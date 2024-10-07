import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, ChevronDown } from "lucide-react";
import Image from "next/image";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { debounce } from 'lodash';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  isLoading: boolean;
  onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  selectedModel: string;
  setSelectedModel: (modelId: string) => void;
  models: { id: string; name: string }[];
}

const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  isLoading,
  onSubmit,
  selectedModel,
  setSelectedModel,
  models,
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const setTextareaRef = useCallback((textarea: HTMLTextAreaElement | null) => {
    if (textarea) {
      textarea.style.height = '56px';
      textareaRef.current = textarea;
    }
  }, []);

  const adjustTextareaHeight = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '56px';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 300)}px`;
    }
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [input, adjustTextareaHeight]);

  const handleSubmit = useCallback((e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (input.trim()) {
      onSubmit(e);
      if (textareaRef.current) {
        textareaRef.current.style.height = '56px'; // Reset textarea height after submission
      }
    }
  }, [input, onSubmit]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  }, [handleSubmit]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    adjustTextareaHeight();
  }, [setInput, adjustTextareaHeight]);

  const selectedModelName = models.find((m) => m.id === selectedModel)?.name;

  return (
    <form onSubmit={handleSubmit} className={`
      flex flex-col w-full relative bg-background border rounded-xl
      transition-all duration-200 ease-in-out
      ${isFocused ? 'ring-2 ring-ring ring-offset-2' : 'shadow-sm'}
    `}>
      <Textarea
        ref={setTextareaRef}
        value={input}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onFocus={() => {
          setIsFocused(true);
          adjustTextareaHeight();
        }}
        onBlur={() => {
          setIsFocused(false);
          if (textareaRef.current) {
            textareaRef.current.style.height = '56px';
          }
        }}
        placeholder="Type your message here..."
        disabled={isLoading}
        className="
          flex-grow resize-none py-4 px-4 min-h-[56px] max-h-[300px] overflow-y-auto
          border-none focus:ring-0 scrollbar-thin scrollbar-thumb-gray-300
          scrollbar-track-transparent hover:scrollbar-thumb-gray-400
          focus:scrollbar-thumb-gray-400 transition-colors
        "
      />
      <div className="flex justify-between items-center p-3 bg-muted/50 rounded-b-xl">
        <Image
          src="/claude-icon.svg"
          alt="Claude Icon"
          width={0}
          height={14}
          className="w-auto h-[14px] mt-1"
        />
        <div className="flex items-center space-x-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="text-muted-foreground">
                {selectedModelName}
                <ChevronDown className="ml-2 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              {models.map(({ id, name }) => (
                <DropdownMenuItem key={id} onSelect={() => setSelectedModel(id)}>
                  {name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="gap-2"
            size="sm"
          >
            {isLoading ? (
              <div className="animate-spin h-5 w-5 border-t-2 border-white rounded-full" />
            ) : (
              <>
                Send Message
                <Send className="h-4 w-4" />
              </>
            )}
          </Button>
        </div>
      </div>
    </form>
  );
};

export default ChatInput;
