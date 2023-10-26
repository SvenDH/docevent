from string import ascii_uppercase
from dataclasses import dataclass

from lark import Lark, Transformer, v_args, Discard


grammar = """
root: actions
actions: touch | rm | mkdir | rmdir | chdir | cat
touch: C "reate a new file named \\"" FILENAME "\\""
rm: D "elete the file \\"" FILENAME "\\""
mkdir: C "reate a new directory named \\"" DIRNAME "\\""
rmdir: D "elete the directory \\"" DIRNAME "\\""
chdir: N "avigate to directory \\"" DIRNAME "\\""
cat: R "ead the file \\"" FILENAME "\\""
{base}
"""


@dataclass
class CreateFileAction:
    name: str


@dataclass
class DeleteFileAction:
    name: str


@dataclass
class CreateDirAction:
    name: str


@dataclass
class DeleteDirAction:
    name: str


@dataclass
class NavigateAction:
    name: str


@dataclass
class ReadAction:
    name: str


class DropLetters(Transformer):
    def __init__(self, visit_tokens: bool = True) -> None:
        for l in ascii_uppercase:
            setattr(self, l, self._drop_token.__func__)
        super().__init__(visit_tokens)

    def _drop_token(self):
        return Discard


@v_args(inline=True)
class NumberTransformer(Transformer):
    def smallnumber(self, item):
        try:
            return int(item)
        except:
            return str(item)
    def number(self, item):
        return item
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
class StringTransformer(Transformer):
    def FILENAME(self, *items):
        return ".".join(items)
    def DIRNAME(self, *items):
        return "/".join(items)


@v_args(inline=True)
class ActionTransformer(Transformer):
    def root(self, item):
        return item
    def actions(item):
        return item
    def touch(self, item):
        return CreateFileAction(name=item)
    def rm(self, item):
        return DeleteFileAction(name=item)
    def mkdir(self, item):
        return CreateDirAction(name=item)
    def rmdir(self, item):
        return DeleteDirAction(name=item)
    def chdir(self, item):
        return NavigateAction(name=item)
    def cat(self, item):
        return ReadAction(name=item)


class Parser:
    def __init__(self, transformers: list[Transformer], debug: bool = True) -> None:
        base = open("grammars/base.lark", "r").read()
        self.grammar = grammar.format(base=base)
        self.lark = Lark(self.grammar, start="root", debug=debug)
        self.transformers = transformers
        self.debug = debug

    def parse(self, txt: str):
        t = self.lark.parse(txt)
        for tf in self.transformers:
            t = tf.transform(t)
        return t


class State:
    def __init__(self, root: str = "data") -> None:
        self.root = root

    def do_action(self, )


parser = Parser([DropLetters(), NumberTransformer(), StringTransformer(), ActionTransformer()])


action = 'Create a new file named "file.txt"'
t = parser.parse(action)

print(t)