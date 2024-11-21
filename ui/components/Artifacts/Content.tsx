import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HtmlPreview from './HtmlPreview';
import Logs from './Logs';

interface ContentProps {
  showPreview: boolean;
  showLogs: boolean;
  activeArtifactData: ArtifactContent | undefined;
  logEntries: {[key: string]: OutputEntry[]};
  artifacts: ArtifactContent[];
  activeArtifact: string | null;
  renderArtifact: (artifact: ArtifactContent) => React.ReactNode;
}

export default function ArtifactContent({
  showPreview,
  showLogs,
  activeArtifactData,
  logEntries,
  artifacts,
  activeArtifact,
  renderArtifact
}: ContentProps) {
  const previewableLanguages = ['html', 'javascript', 'typescript', 'tsx'];

  const renderContent = () => {
    if (showPreview && activeArtifactData && previewableLanguages.includes(activeArtifactData.language?.toLowerCase() || '')) {
      return <HtmlPreview content={activeArtifactData.content} language={activeArtifactData.language || ''} />;
    }

    if (showLogs && activeArtifactData) {
      return <Logs entries={logEntries[activeArtifactData.id] || []} />;
    }

    return activeArtifact && renderArtifact(artifacts.find(a => a.id === activeArtifact)!);
  };

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={showPreview ? 'preview' : showLogs ? 'logs' : 'artifact'}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.2 }}
        className="flex-grow"
      >
        {renderContent()}
      </motion.div>
    </AnimatePresence>
  );
}
