'use client';

import React, { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import SimpleEditor from "./SimpleEditor";

interface EditModalProps {
  isOpen: boolean;
  onClose: () => void;
  content: string;
  onSave: (newContent: string) => void;
}

const EditModal: React.FC<EditModalProps> = ({
  isOpen,
  onClose,
  content,
  onSave,
}) => {
  const [editedContent, setEditedContent] = useState(content);

  useEffect(() => {
    if (isOpen) {
      setEditedContent(content);
    }
  }, [isOpen, content]);

  const handleSave = () => {
    onSave(editedContent);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] w-[850px] max-h-[95vh] flex flex-col p-6">
        <DialogHeader className="mb-4">
          <DialogTitle>Edit Message</DialogTitle>
        </DialogHeader>
        <div className="flex-grow overflow-hidden mb-4">
          <div className="h-[calc(80vh-120px)] w-full">
            <SimpleEditor
              key={isOpen ? 'open' : 'closed'}
              value={editedContent}
              onChange={(value) => setEditedContent(value || '')}
              language="markdown"
            />
          </div>
        </div>
        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSave}>Save</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default EditModal;