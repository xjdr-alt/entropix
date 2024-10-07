import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area"

interface TextPreviewProps {
  content: string;
}

export default function TextPreview({ content }: TextPreviewProps) {
  return (
    <ScrollArea className="h-full bg-background rounded-lg border overflow-hidden p-4">
      <pre className="text-sm text-foreground whitespace-pre-wrap">{content}</pre>
    </ScrollArea>
  );
}