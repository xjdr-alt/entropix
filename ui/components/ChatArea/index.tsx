"use client";

import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { X, Code, SlidersHorizontal } from "lucide-react";
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import WelcomeMessage from './WelcomeMessage';
import Artifacts from '../Artifacts';
import EditModal from './EditModal';
import ArtifactSidebar from '../Artifacts/Sidebar';
import { motion, AnimatePresence } from 'framer-motion';
import { Message, Model, ArtifactContent } from "@/types/chat";

function ChatArea() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("/sax/test/pdex_llama3_70b");
  const [showArtifacts, setShowArtifacts] = useState(false);
  const [editingMessage, setEditingMessage] = useState<Message | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactContent[]>([
    { id: '1', type: 'code', content: '# Your Python code here\nprint("Hello, World!")', language: 'python', name: 'Initial Code' },
  ]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const models: Model[] = [
    { id: "/sax/test/pdex_llama3_8b", name: "Spark" },
    { id: "/sax/test/pdex_llama3_70b", name: "Atlas" },
  ];

  const [assistantMessage, setAssistantMessage] = useState<Message | null>(null);

  const [showArtifactSidebar, setShowArtifactSidebar] = useState(false);

  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      }, 100);
    }
  }, [messages]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (input.trim() === "") return;

    setIsLoading(true);

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input,
    };

    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput("");

    try {
      // Simulating API call
      const response = await new Promise<string>((resolve) => {
        setTimeout(() => {
          resolve(JSON.stringify({
            response: `This is a simulated response to: "${input}"`,
            thinking: "Thinking process...",
            user_mood: "neutral"
          }));
        }, 1000);
      });

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response,
      };

      setAssistantMessage(assistantMessage);
      setMessages(prevMessages => [...prevMessages, assistantMessage]);

      // Simulating artifact generation
      const newArtifact: ArtifactContent = {
        id: crypto.randomUUID(),
        type: 'code',
        content: `# Generated code\nprint("Response to: ${input}")`,
        language: 'python',
        name: 'Generated Code'
      };
      setArtifacts(prevArtifacts => [...prevArtifacts, newArtifact]);
    } catch (error) {
      console.error("Error fetching chat response:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEditClick = (message: Message) => {
    setEditingMessage(message);
  };

  const handleSaveEdit = (newContent: string) => {
    if (editingMessage) {
      setMessages((prevMessages) =>
        prevMessages.map((msg) =>
          msg.id === editingMessage.id ? { ...msg, content: newContent } : msg
        )
      );
    }
    setEditingMessage(null);
  };

  const handleArtifactChange = (id: string, value: string) => {
    setArtifacts(prevArtifacts =>
      prevArtifacts.map(artifact =>
        artifact.id === id ? { ...artifact, content: value } : artifact
      )
    );
  };

  const handleRunCode = (code: string) => {
    console.log("Running code:", code);
    // Implement code execution logic here
    // After execution, you might want to add a new log artifact with the result
    const newLogArtifact: ArtifactContent = {
      id: crypto.randomUUID(),
      type: 'log',
      content: `Execution result of: ${code}`,
      name: 'Execution Log'
    };
    setArtifacts(prev => [...prev, newLogArtifact]);
  };

  const handleDeleteArtifact = (id: string) => {
    setArtifacts(prevArtifacts => prevArtifacts.filter(artifact => artifact.id !== id));
  };

  const artifactsVariants = {
    hidden: { x: "100%", opacity: 0 },
    visible: { x: 0, opacity: 1 }
  };

  return (
    <div className="flex flex-1 mb-4 mx-4 space-x-4 relative h-full">
      <div
        className="flex flex-col mx-auto"
        style={{ width: showArtifacts ? '50%' : '100%', maxWidth: showArtifacts ? '100%' : '48rem' }}
      >
        <Card className="flex-1 flex flex-col shadow-none border-none h-full w-full">
          <CardContent className="flex-1 flex flex-col overflow-hidden pt-4 px-4 pb-0">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 ? (
                <WelcomeMessage />
              ) : (
                <MessageList
                  messages={messages}
                  onEditClick={handleEditClick}
                  messagesEndRef={messagesEndRef}
                />
              )}
            </div>
          </CardContent>
          <CardFooter className="p-0 mt-auto">
            <ChatInput
              input={input}
              setInput={setInput}
              isLoading={isLoading}
              onSubmit={handleSubmit}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              models={models}
            />
          </CardFooter>
        </Card>
      </div>

      <AnimatePresence>
        {showArtifacts && (
          <motion.div
            initial="hidden"
            animate="visible"
            exit="hidden"
            variants={artifactsVariants}
            transition={{ duration: 0.4, ease: "easeInOut" }}
            className="w-1/2"
          >
            <Artifacts
              artifacts={artifacts}
              onChange={handleArtifactChange}
              onRun={handleRunCode}
              onDelete={handleDeleteArtifact}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <EditModal
        isOpen={!!editingMessage}
        onClose={() => setEditingMessage(null)}
        content={editingMessage?.content || ''}
        onSave={handleSaveEdit}
      />

      {!showArtifactSidebar && (
        <Button
          onClick={() => setShowArtifacts(!showArtifacts)}
          className="absolute top-4 right-4 z-10"
          size="sm"
          variant="outline"
        >
          {showArtifacts ? <X className="h-4 w-4" /> : <Code className="h-4 w-4" />}
        </Button>
      )}

      <ArtifactSidebar
        isOpen={showArtifactSidebar}
        setIsOpen={setShowArtifactSidebar}
        artifacts={artifacts.map(a => ({
          id: a.id,
          type: a.type === 'image' ? 'image' : 'code',
          name: a.name,
          versions: 1,
        }))}
      />
    </div>
  );
}

export default ChatArea;