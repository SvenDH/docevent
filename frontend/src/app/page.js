'use client';
import {useState, useEffect} from 'react'
import styles from './page.module.css'
import { Container , VStack, HStack, Box, Grid, GridItem, Input, Textarea, Text, Divider, Switch, Alert, ChakraProvider } from '@chakra-ui/react'
import { AddIcon, EditIcon, CheckIcon, DeleteIcon } from '@chakra-ui/icons'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import 'github-markdown-css'

const BASE_URL = (process.env.REACT_APP_BASE_URL || '') + '/api'


function Document() {
  const [file, setFile] = useState('document.txt');
  const [doc, setDoc] = useState('');
  const [dir, setDir] = useState([]);
  const [edit, setEdit] = useState(false);
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState('');
  const [error, setError] = useState(null);
  
  const fetchDir = async (signal = null) => {
    const rsp = await fetch(`${BASE_URL}/list/`, {signal, redirect: 'follow'});
    setDir(await rsp.json());
  };

  const resetPath = () => {
    setPath('');
    setOpen(false);
  }

  const openDoc = path => {
    setEdit(false)
    setFile(path)
  }

  const openEdit = () => {
    setDoc('')
    setEdit(true)
  }

  const closeEdit = () => {
    setDoc('')
    setEdit(false)
  }

  const newDoc = async () => {
    await fetch(`${BASE_URL}/edit/${path}`, {
      method: 'POST', 
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: ''})
    });
    resetPath();
    fetchDir();
  };

  const editDoc = async e => {
    const text = e.target.value;
    setDoc(text);
    const rsp = await fetch(`${BASE_URL}/edit/${file}`, {
      method: 'POST', 
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const res = await rsp.json();
    if (res["status"] !== "OK") {
      const line = text.split("\n")[res["line"]-1].slice(res["col"], res["col"]+30);
      setError(line);
    }
    else {
      setError(null);
    }
  };

  const removeDoc = async path => {
    if (path === file) setFile(null);
    await fetch(`${BASE_URL}/edit/${path}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: null})
    });
    fetchDir();
  };

  useEffect(() => {
    const abortController = new AbortController()
    fetchDir(abortController.signal);
    return () => {abortController.abort()};
  }, []);

  useEffect(() => {
    if (file) {
      const abortController = new AbortController()
      const fetchStream = async () => {
        if (!edit) {
          try {
            const rsp = await fetch(`${BASE_URL}/stream/${file}`, {signal: abortController.signal});
            const reader = rsp.body.getReader();
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              const newText = new TextDecoder().decode(value);
              setDoc(newText.split("\n").slice(-30).join("\n"));
            }
          } catch (e) {
            if (!abortController.signal.aborted) {
              console.log(e)
            }
          }
        }
        else {
          const rsp = await fetch(`${BASE_URL}/stream/${file}`, {signal: abortController.signal});
          const reader = rsp.body.getReader();
          const { value } = await reader.read();
          setDoc(new TextDecoder().decode(value))
          abortController.abort()
        }
      };
      fetchStream();
      if (!edit) return () => {abortController.abort()};
    }
  }, [file, edit]);

  return (
    <Container minW='100%' minH='100vh' bg='gray.800' paddingLeft={20} paddingRight={20}>
      <Box minH={100}></Box>
      <Grid templateColumns='repeat(4, 1fr)' minH='100%' gap={6}>
        <GridItem w='100%' padding={3} bg='gray.900' borderRadius='md'>
          <VStack color='white'>
            {dir.map((t, i) => 
              <HStack w='100%'>
                <Text w='100%' cursor='pointer' onClick={() => openDoc(t)} key={`dir-${i}`}>{t}</Text>
                <DeleteIcon cursor='pointer' onClick={() => removeDoc(t)}/>
              </HStack>)}
            {open ? 
              <HStack>
                <Input value={path} placeholder='File name' size='sm' onChange={e => setPath(e.target.value)}/>
                <CheckIcon onClick={newDoc}/>
                <DeleteIcon onClick={resetPath}/>
              </HStack> : 
              <AddIcon onClick={() => setOpen(true)} cursor='pointer' size='sm' />}
          </VStack>
        </GridItem>
        <GridItem colSpan={2} w='100%' padding={3} bg='gray.900' borderRadius='md'>
          <VStack w='100%' color='white'>
            <Text>{file} {edit ? <CheckIcon onClick={closeEdit}/> : <EditIcon onClick={openEdit}/>}</Text>
            <Divider />
            {edit ?
            <>
              {error && <Alert status='error' size='xs' color='black'>{error}</Alert>}
              <Textarea
                value={doc}
                h={'calc(100vh - 200px)'}
                onChange={e => editDoc(e)}
              />
            </> :
            <Box w='100%' align='left'>
              <div className='markdown-body' styles={{}}>
                <Markdown remarkPlugins={[remarkGfm]}>{doc.replace(/\n/gi, '  \n')}</Markdown>
              </div>
            </Box>}
          </VStack>
        </GridItem>
        <GridItem w='100%' />
      </Grid>
    </Container>
  )
}

export default function Home() {
  return (
    <ChakraProvider>
      <Document/>
    </ChakraProvider>
  )
}
