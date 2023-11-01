import re
import json
import asyncio
from typing import ClassVar
from pathlib import Path
from string import ascii_uppercase
from dataclasses import dataclass
from typing import AsyncIterable

import uvicorn
import aiofiles
import aiofiles.os
from pydantic import BaseModel
from aiostream import stream, pipable_operator, streamcontext
from watchfiles import awatch
from lark import Lark, Transformer, v_args, Discard, UnexpectedCharacters
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_cpp import Llama, LlamaGrammar
from langchain.llms.openai import OpenAI


grammar = """
root: action (sep action)*
action: ( {actionoptions} ) (" " cond)?
cond: ( {condoptions} )
{actions}

text: string | ("'" jsonvalue "'")
textorit: text | "it"
textorfile: text | ("file " FILENAME)
textoritorfile: textorit | ("file " FILENAME)
theendof: "the end of "
formatted: " formatted"
sep: "\\n"+

{base}
"""

REGEX = "\/((?![*+?])(?:[^\r\n\[/\\]|\\.|\[(?:[^\r\n\]\\]|\\.)*\])+)\/((?:g(?:im?|mi?)?|i(?:gm?|mg?)?|m(?:gi?|ig?)?)?)"


@dataclass
class Action:
    def pipe(self, state, pipe):
        if pipe is not None:
            @pipable_operator
            async def runner(source: AsyncIterable[str]):
                async with streamcontext(source) as streamer:
                    async for line in streamer:
                        state.line = line
                        if line is not None:
                            async for r in self.run(state):
                                yield r
                        yield None
            return pipe | runner.pipe()
        return stream.iterate(self.run(state))

    async def run(self, state):
        pass


class State:
    def __init__(self, root: str = "data") -> None:
        self.root = Path(root).resolve()
        self.ch = self.root
        self.files = {}
        self.line = None
        self.ctx = {}

    def changedir(self, path):
        self.ch = (self.ch / Path(path)).resolve()
        if str(self.root) not in str(self.ch):
            self.ch = self.root

    def fullpath(self, path):
        return self.ch / path
    
    def format(self, line: str) -> str:
        if line is not None:
            return line.format(**self.ctx)
        return None
    
    def run(self, path: str, action: Action):
        logger = WriteAction(path=str(path).replace(".task", ".log"), overwrite=False)
        return logger.pipe(self, action.pipe(self, None))


@dataclass
class Cond(Action):
    def pipe(self, state: State, pipe):
        @pipable_operator
        async def runner(source: AsyncIterable[str]):
            async with streamcontext(source) as streamer:
                async for line in streamer:
                    if self.match(state, line):
                        yield line
        return pipe | runner.pipe()
    
    def match(self, state: State, line: str):
        return True


@dataclass
class PrintAction(Action):
    def pipe(self, state: State, pipe):
        @pipable_operator
        async def printrunner(source: AsyncIterable[str]):
            async with streamcontext(source) as streamer:
                async for line in streamer:
                    if line is not None:
                        print(line)
                    yield None
        return pipe | printrunner.pipe()


@dataclass
class FileAction(Action):
    path: Path | None = None


@dataclass
class CreateFileAction(FileAction):
    pattern: ClassVar[str] = 'C "reate file \\"" FILENAME "\\"" (" with " text)?'

    ifnotexist: bool = True
    text: str = ""

    async def run(self, state: State):
        path = state.fullpath(self.path)
        state.files[path] = await aiofiles.open(path, "w+" if self.text else "a+")
        if self.text:
            await state.files[path].write(self.text)
        yield None


@dataclass
class DeleteFileAction(FileAction):
    pattern: ClassVar[str] = 'D "elete file \\"" FILENAME "\\""'

    async def run(self, state: State):
        yield await aiofiles.os.unlink(state.fullpath(self.path))


@dataclass
class CreateDirAction(FileAction):
    pattern: ClassVar[str] = 'C "reate directory \\"" DIRNAME "\\""'

    ifnotexist: bool = True

    async def run(self, state: State):
        yield await aiofiles.os.mkdir(state.fullpath(self.path))


@dataclass
class DeleteDirAction(FileAction):
    pattern: ClassVar[str] = 'D "elete directory \\"" DIRNAME "\\""'

    async def run(self, state: State):
        yield await aiofiles.os.rmdir(state.fullpath(self.path))


@dataclass
class NavigateAction(FileAction):
    pattern: ClassVar[str] = 'N "avigate to \\"" DIRNAME "\\""'

    async def run(self, state: State):
        yield state.changedir(self.path)


@dataclass
class ListDirAction(FileAction):
    pattern: ClassVar[str] = 'L "ist \\"" DIRNAME "\\""'

    filesonly: bool = False

    async def run(self, state: State):
        for f in await aiofiles.os.scandir(state.fullpath(self.path)):
            if not self.filesonly or f.is_file():
                yield f.name


@dataclass
class ReadAction(FileAction):
    pattern: ClassVar[str] = 'R "ead \\"" FILENAME "\\""'

    lines: bool = False
    follow: bool = False
    head: int = -1
    tail: int = -1

    async def run(self, state: State):
        file = state.fullpath(self.path)
        async with aiofiles.open(file) as f:
            if self.tail >= 0:
                await f.seek(-self.tail, 2)
            if self.lines:
                async for line in f:
                    yield line.strip("\n\r")
            else:
                yield await f.read()
            if self.follow:
                async for _ in awatch(file, yield_on_timeout=True):
                    lines = await f.read()
                    if self.lines:
                        for line in lines.strip("\n\r").split("\n"):
                            if line:
                                yield line
                    elif lines:
                        yield lines


@dataclass
class FollowAction(ReadAction):
    pattern: ClassVar[str] = 'F "ollow \\"" FILENAME "\\""'

    follow: bool = True


@dataclass
class WriteAction(FileAction):
    pattern: ClassVar[str] = 'W "rite " textorit formatted? (" to " theendof? "\\"" FILENAME "\\"")?'

    text: str | None = None
    line: bool = True
    overwrite: bool = True
    format: bool = False

    async def run(self, state: State):
        line = self.text or state.line or ""
        if self.format:
            line = state.format(line)
        line += ("\n" if self.line else "")
        if self.path is None:
            yield line
        else:
            async with aiofiles.open(state.fullpath(self.path), "w+" if self.overwrite else "a+") as f:
                await f.write(line)
                await f.flush()
                yield None


@dataclass
class StoreAction(FileAction):
    pattern: ClassVar[str] = 'S "tore " textoritorfile (" as \\"" WORD "\\"")?'
    
    text: str | None = None
    var: str | None = None

    async def run(self, state: State):
        if self.path is not None:
            async with aiofiles.open(state.fullpath(self.path)) as f:
                data = await f.read()
        elif self.text is not None:
            data = self.text
        else:
            data = state.line
        if self.var is not None:
            state.ctx[state.format(self.var)] = data
        else:
            try:
                state.ctx.update(**json.loads(data))
            except:
                pass
        yield data


@dataclass
class AskAction(FileAction):
    pattern: ClassVar[str] = 'A "sk " textoritorfile'
    
    text: str | None = None

    async def run(self, state: State):
        if self.path is not None:
            async with aiofiles.open(state.fullpath(self.path)) as f:
                data = await f.read()
        elif self.text is not None:
            data = self.text
        else:
            data = state.line
        data = state.format(data)
        llm = OpenAI()
        yield await llm.apredict(data)


@dataclass
class ReplaceAction(FileAction):
    pattern: ClassVar[str] = 'R "eplace " text " with " textorfile'
    
    text: str | None = None
    replace: str | None = None

    def set_field(self, state: State, data, text: str):
        idx = state.format(self.text)
        try:
            data[int(idx)] = text
        except:
            data[idx] = text
        return data

    async def run(self, state: State):
        if self.path is not None:
            async with aiofiles.open(state.fullpath(self.path)) as f:
                text = await f.read()
        else:
            text = self.replace
        text = state.format(text)
        try:
            data = json.loads(state.line)
            data = json.dumps(self.set_field(state, data, text))
        except:
            data = state.line.split(" ")
            data = " ".join(self.set_field(state, data, text))
        yield data


@dataclass
class Match(Cond):
    pattern: ClassVar[str] = 'W "hen it contains " text'

    text: str
    
    def match(self, state: State, line: str):
        return state.format(self.text) in line


@dataclass
class Conditional(Action):
    action: Action
    cond: Cond

    def pipe(self, state: State, pipe):
        pipe = self.cond.pipe(state, pipe)
        return self.action.pipe(state, pipe)


@dataclass
class Composed(Action):
    actions: list[Action]

    def pipe(self, state: State, pipe = None):
        for action in self.actions:
            pipe = action.pipe(state, pipe)
        return pipe


class DropLetters(Transformer):
    def __init__(self, visit_tokens: bool = True) -> None:
        for l in ascii_uppercase:
            setattr(self, l, self._drop_token.__func__)
        super().__init__(visit_tokens)

    def _drop_token(self):
        return Discard
    def sep(self, _):
        return Discard
    def ws(self, _):
        return Discard


@v_args(inline=True)
class NumberTransformer(Transformer):
    def smallnumber(self, item):
        return int(item)
    def a(self):
        return 1
    def one(self):
        return 1
    def two(self):
        return 2
    def three(self):
        return 3
    def four(self):
        return 4
    def five(self):
        return 5
    def six(self):
        return 6
    def seven(self):
        return 7
    def eight(self):
        return 8
    def nine(self):
        return 9
    def ten(self):
        return 10


@v_args(inline=True)
class JsonTransformer(Transformer):
    def jsonvalue(self, item):
        return item
    def object(self, *items):
        return {items[i]: items[i+1] for i in range(0, len(items), 2)}
    def array(self, *items):
        return list(items)
    def string(self, *items):
        return "".join([str(i) for i in items])
    def number(self, item):
        if int(item) == float(item):
            return int(item)
        return float(item)
    def TRUE(self):
        return True
    def FALSE(self):
        return False
    def NULL(self):
        return None


@v_args(inline=True)
class StringTransformer(Transformer):
    def FILENAME(self, *items):
        return Path(".".join(items))
    def DIRNAME(self, *items):
        return Path("/".join(items))
    def WORD(self, item):
        return item
    def REGEX(self, item):
        return re.compile(item)
    def text(self, item):
        if isinstance(item, dict):
            return "{" + json.dumps(item) + "}"
        return item
    def textorit(self, *args):
        if (args):
            return args[0]
        return None
    def textorfile(self, item):
        return item
    def textoritorfile(self, item):
        return item
    def formatted(self):
        return "formatted"
    def endof(self):
        return "endof"


TERMINAL_ACTION = PrintAction()


@v_args(inline=True)
class ActionTransformer(Transformer):
    def root(self, *items):
        return Composed(actions=list(items))
    def action(self, item, *args):
        if args:
            return Conditional(action=item, cond=args[0])
        return item
    def cond(self, item):
        return item
    def createfileaction(self, item, *args):
        return CreateFileAction(path=item, ifnotexist=len(args) > 0, text=args[0] if args else "")
    def deletefileaction(self, item):
        return DeleteFileAction(path=item)
    def creatediraction(self, item):
        return CreateDirAction(path=item, ifnotexist=True)
    def deletediraction(self, item):
        return DeleteDirAction(path=item)
    def navigateaction(self, item):
        return NavigateAction(path=item)
    def readaction(self, item):
        return ReadAction(path=item, lines=False, follow=False)
    def listdiraction(self, item):
        return ListDirAction(path=item)
    def followaction(self, item):
        return FollowAction(path=item, lines=True, follow=True, tail=0)
    def writeaction(self, text, *args):
        if len(args) > 1:
            path = None
            format = args[0] == "formatted"
            overwrite = args[-2] != "endof"
            if isinstance(args[-1], Path):
                path = args[-1]
            return WriteAction(path=path, format=format, text=text, overwrite=overwrite)
        return WriteAction(path=args[0] if args else None, text=text, overwrite=True)
    def storeaction(self, text, *args):
        var = None
        if args:
            var = args[0]
        if isinstance(text, Path):
            return StoreAction(path=text, var=var)
        elif text is not None:
            return StoreAction(text=text, var=var)
        return StoreAction(var=var)
    def askaction(self, text):
        if isinstance(text, Path):
            return AskAction(path=text)
        elif text is not None:
            return AskAction(text=text)
        return AskAction()
    def replaceaction(self, text, other):
        if isinstance(other, Path):
            return ReplaceAction(text=text, path=other)
        return ReplaceAction(text=text, replace=other)
    def match(self, item):
        return Match(text=item)


class Parser:
    def __init__(self, actions: list, conditions: list, transformers: list, debug: bool = True) -> None:
        base = open("grammars/base.lark", "r").read()
        actionstr = ""
        for action in actions + conditions:
            actionstr += f"\n{action.__name__.lower()}: {action.pattern}"
        self.grammar = grammar.format(
            base=base,
            actionoptions=" | ".join([a.__name__.lower() for a in actions]),
            condoptions=" | ".join([c.__name__.lower() for c in conditions]),
            actions=actionstr
        )
        self.lark = Lark(self.grammar, start="root", debug=debug)
        self.transformers = [t() for t in transformers]
        self.debug = debug

    def parse(self, txt: str):
        t = self.lark.parse(txt)
        for tf in self.transformers:
            t = tf.transform(t)
        return t


class Generator:
    def __init__(self, transformers: list[Transformer], model_path: str, temperature: float = 1.0, debug: bool = True) -> None:
        self.temperature = temperature
        self.model = Llama(
            model_path,
            seed=-1,
            n_ctx=512,
            n_gpu_layers=128,
            n_batch=512,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,
        )
        self.parser = Parser(transformers=transformers, debug=debug)

    def generate(self, prompt: str):
        g = self.parser.grammar
        g = g.replace(" /", " ").replace("/ ", " ").replace("/\n", "\n").replace("\n|", " |").replace(": ", " ::= ")
        g = LlamaGrammar.from_string(g)
        result = self.model(
            prompt=prompt,
            grammar=g,
            temperature=self.temperature,
            max_tokens=256
        )
        return result["choices"][0]["text"]


actions = [CreateFileAction, DeleteFileAction, CreateDirAction, DeleteDirAction, NavigateAction, ListDirAction, ReadAction, FollowAction, WriteAction, StoreAction, AskAction, ReplaceAction]
conditions = [Match]
transformers = [DropLetters, NumberTransformer, JsonTransformer, StringTransformer, ActionTransformer]

parser = Parser(actions, conditions, transformers)


#generator = Generator(transformers, "models/mistral-7b-v0.1.Q3_K_L.gguf")
#prompt = 'Implement eventsourcing and cqrs on "event.log".'
#t = generator.generate(prompt)
#print(t)


tasks = {}


async def remover(path, task):
    await task
    del tasks[path]


async def register_task(state: State, path: str):
    path = state.fullpath(path)
    try:
        async with aiofiles.open(path) as f:
            action = parser.parse(await f.read())
        if path in tasks:
            tasks[path].cancel()
        tasks[path] = asyncio.create_task(remover(path, state.run(path, action)))
    except UnexpectedCharacters as e:
        print(e._context)
        return {
            "status": "ERR",
            "line": e.line,
            "col": e.column
        }
    except Exception as e:
        print(e)
        pass
    return {"status": "OK"}


async def tick():
    from datetime import datetime
    async with aiofiles.open("data/time.log", "a+") as f:
        while True:
            await f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            await f.flush()
            await asyncio.sleep(1.0)


async def register():
    asyncio.create_task(tick())
    state = State()
    action = ListDirAction(".")
    async for entry in action.run(state):
        if entry.endswith(".task"):
            await register_task(state, entry)


app = FastAPI(on_startup=[register])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def streamgenerator(file: str, tail: int = 0, follow: bool = True):
    async with aiofiles.open(file, "r+") as f:
        if tail < 0:
            await f.seek(0, 2)
            await f.seek(max(await f.tell() + tail, 0), 0)
        lastdata = await f.read()
        yield lastdata + "\n"
        if follow:
            async for _ in awatch(file, yield_on_timeout=True):
                if tail < 0:
                    await f.seek(0, 2)
                    await f.seek(max(await f.tell() + tail, 0), 0)
                else:
                    await f.seek(0, 0)
                data = await f.read()
                if data is not None and data != lastdata:
                    yield data + "\n"
                    lastdata = data


class FileEdit(BaseModel):
    text: str | None = None
    overwrite: bool = True


@app.get("/api/stream/{path}")
def streamfile(path: str, tail: int = 0, follow: bool = True):
    state = State()
    return StreamingResponse(streamgenerator(state.fullpath(path), tail, follow))


@app.get("/api/list/")
@app.get("/api/list/{path}")
async def streamfile(path: str = "."):
    state = State()
    action = ListDirAction(path)
    return [entry async for entry in action.run(state)]


@app.post("/api/edit/{path}")
async def streamfile(path: str, edit: FileEdit):
    state = State()
    if edit.text is None:
        action = DeleteFileAction(path)
    else:
        action = WriteAction(path, edit.text, format=False, line=False, overwrite=edit.overwrite)
    async for _ in action.run(state):
        pass
    if edit.text is not None and path.endswith(".task"):
        return await register_task(state, path)
    return {"status": "OK"}


uvicorn.run(app)
