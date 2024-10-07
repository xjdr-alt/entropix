import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import TextEditor from '../TextEditor';

interface EditorPanelProps {
  content: string;
  onChange: (value: string | undefined) => void;
}

const EditorPanel: React.FC<EditorPanelProps> = ({ content, onChange }) => {
  return (
    <Card className="w-1/2 flex flex-col shadow-none border-none">
      <CardContent className="flex-1 relative p-0">
        <TextEditor
          content={content}
          onChange={onChange}
          language="python"
        />
      </CardContent>
    </Card>
  );
};

export default EditorPanel;
