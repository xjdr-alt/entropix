import React from 'react';
import { motion } from 'framer-motion';
import { Globe, Terminal } from 'lucide-react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { cn } from '@/library/utils';

interface TabsProps {
  artifacts: ArtifactContent[];
  activeArtifact: string | null;
  setActiveArtifact: (id: string) => void;
  showPreview: boolean;
  setShowPreview: (show: boolean) => void;
  showLogs: boolean;
  setShowLogs: (show: boolean) => void;
  isPreviewable: (artifact: ArtifactContent | undefined) => boolean;
}

export default function ArtifactTabs({
  artifacts,
  activeArtifact,
  setActiveArtifact,
  showPreview,
  setShowPreview,
  showLogs,
  setShowLogs,
  isPreviewable
}: TabsProps) {
  const activeArtifactObj = artifacts.find(a => a.id === activeArtifact);

  return (
    <div className="flex items-center mb-4">
      <ScrollArea className="flex-grow max-w-[calc(100%-320px)]" orientation="horizontal">
        <div className="flex space-x-2">
          {artifacts.map(({ id, name }) => (
            <ArtifactTab
              key={id}
              id={id}
              name={name}
              isActive={id === activeArtifact}
              onClick={() => setActiveArtifact(id)}
            />
          ))}
        </div>
      </ScrollArea>
      <div className="flex items-center space-x-2 ml-auto mr-12">
        <ToggleButton
          icon={<Terminal className="h-4 w-4" />}
          label={showLogs ? 'Editor' : 'Logs'}
          onClick={() => setShowLogs(!showLogs)}
        />
        {isPreviewable(activeArtifactObj) && (
          <ToggleButton
            icon={<Globe className="h-4 w-4" />}
            label={showPreview ? 'Editor' : 'Preview'}
            onClick={() => setShowPreview(!showPreview)}
          />
        )}
      </div>
    </div>
  );
}

const ArtifactTab = ({ id, name, isActive, onClick }) => (
  <motion.button
    onClick={onClick}
    className={cn(
      "px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 ease-in-out",
      isActive
        ? "bg-primary text-primary-foreground shadow-lg"
        : "bg-muted text-muted-foreground hover:bg-muted/80"
    )}
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
  >
    <div className="flex items-center space-x-2">
      <span>{name}</span>
    </div>
  </motion.button>
);

const ToggleButton = ({ icon, label, onClick }) => (
  <Button
    variant="ghost"
    size="sm"
    onClick={onClick}
    className="flex items-center space-x-2"
  >
    {icon}
    <span>{label}</span>
  </Button>
);
