import React from 'react';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Pencil } from "lucide-react";
import MessageContent from './MessageContent';

interface Message {
  id: string;
  role: string;
  content: string;
}

interface MessageListProps {
  messages: Message[];
  onEditClick: (message: Message) => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
}

const MessageList: React.FC<MessageListProps> = ({ messages, onEditClick, messagesEndRef }) => {
  return (
    <div className="space-y-4 flex flex-col items-center">
      {messages.map((message) => (
        <div key={message.id} className="w-full max-w-2xl">
          <div className={`flex items-start ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            {message.role === "assistant" && (
              <Avatar className="w-8 h-8 mr-2 border">
                <AvatarImage src="/ant-logo.svg" alt="AI Assistant Avatar" />
                <AvatarFallback>AI</AvatarFallback>
              </Avatar>
            )}
            <div className={`p-3 rounded-md text-sm w-full relative group ${
              message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted border"
            }`}>
              <MessageContent content={message.content} role={message.role} />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onEditClick(message)}
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <Pencil className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      ))}
      <div ref={messagesEndRef} style={{ height: "1px" }} />
    </div>
  );
};

export default MessageList;
