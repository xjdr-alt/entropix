import React from 'react';
import { Check, Copy, RotateCcw, Play, Trash2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface ActionsProps {
  activeArtifactData: ArtifactContent | undefined;
  copied: boolean;
  setCopied: (copied: boolean) => void;
  copyToClipboard: () => void;
  resetContent: () => void;
  runCode: () => void;
  onDelete: (id: string) => void;
  activeArtifact: string | null;
}

export default function ArtifactActions({
  activeArtifactData,
  copied,
  setCopied,
  copyToClipboard,
  resetContent,
  runCode,
  onDelete,
  activeArtifact
}: ActionsProps) {
  const actions = [
    { icon: copied ? Check : Copy, onClick: copyToClipboard, tooltip: copied ? 'Copied!' : 'Copy content' },
    { icon: RotateCcw, onClick: resetContent, tooltip: 'Reset content' },
    { icon: Play, onClick: runCode, tooltip: 'Run code', disabled: activeArtifactData?.type !== 'code' },
    { icon: Trash2, onClick: () => activeArtifact && onDelete(activeArtifact), tooltip: 'Delete artifact' },
  ];

  return (
    <div className="p-2 border-t flex justify-end space-x-2">
      {actions.map(({ icon: Icon, onClick, tooltip, disabled }, index) => (
        <TooltipProvider key={index}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClick}
                className="bg-background/50 backdrop-blur-sm"
                disabled={disabled}
              >
                <Icon className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ))}
    </div>
  );
}

