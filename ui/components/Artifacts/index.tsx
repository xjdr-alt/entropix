'use client';

import React, { useState, useEffect } from 'react'
import { useTheme } from 'next-themes'
import ArtifactTabs from './Tabs'
import ArtifactContent from './Content'
import ArtifactActions from './Actions'
import ArtifactSidebar from './Sidebar'
import Code from './Code'
import TextPreview from './TextPreview'
import usePyodide from '@/hooks/usePyodide'

interface ArtifactContent {
  id: string;
  type: 'code' | 'html' | 'text' | 'log';
  content: string;
  language?: string;
  name: string;
}

interface ArtifactsProps {
  artifacts: ArtifactContent[];
  onChange: (id: string, value: string) => void;
  onRun: (id: string, code: string) => void;
  onDelete: (id: string) => void;
  selectedArtifact: ArtifactContent | null;
}

const mockArtifacts: ArtifactContent[] = [
  {
    id: 'python1',
    type: 'code',
    content: `
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
    `,
    language: 'python',
    name: 'Python Greeting'
  },
  {
    id: 'tsx1',
    type: 'code',
    content: `
import React from 'react';

interface GreetingProps {
  name: string;
}

const Greeting: React.FC<GreetingProps> = ({ name }) => {
  return <h1>Hello, {name}!</h1>;
};

export default function App() {
  return <Greeting name="World" />;
}
    `,
    language: 'tsx',
    name: 'TSX Greeting'
  }
];

export default function Artifacts({
  artifacts = mockArtifacts,
  onChange,
  onRun,
  onDelete,
  selectedArtifact,
}: ArtifactsProps) {
  const [activeArtifact, setActiveArtifact] = useState<string | null>(selectedArtifact?.id || artifacts[0]?.id || null);
  const [showPreview, setShowPreview] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [logEntries, setLogEntries] = useState<{[key: string]: OutputEntry[]}>({});
  const [copied, setCopied] = useState(false);
  const { resolvedTheme } = useTheme();
  const { pyodide } = usePyodide();

  const activeArtifactData = artifacts.find(a => a.id === activeArtifact);

  const isPreviewable = (artifact: ArtifactContent | undefined) =>
    artifact && ['html', 'javascript', 'typescript', 'tsx'].includes(artifact.language?.toLowerCase() || '');

  const copyToClipboard = () => {
    if (activeArtifactData) {
      navigator.clipboard.writeText(activeArtifactData.content)
        .then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        });
    }
  };

  const runCode = async () => {
    if (activeArtifactData?.type === 'code') {
      if (activeArtifactData.language === 'python' && pyodide) {
        try {
          // Capture console output
          let output = '';
          const originalConsoleLog = console.log;
          console.log = (...args) => {
            output += args.join(' ') + '\n';
            originalConsoleLog.apply(console, args);
          };

          const result = await pyodide.runPython(activeArtifactData.content);
          
          // Restore original console.log
          console.log = originalConsoleLog;

          setLogEntries(prev => ({
            ...prev,
            [activeArtifactData.id]: [
              ...(prev[activeArtifactData.id] || []),
              {
                id: crypto.randomUUID(),
                timestamp: new Date().toISOString(),
                content: output.trim(),
                type: 'output'
              },
              // Only add the result if it's not undefined
              ...(result !== undefined ? [{
                id: crypto.randomUUID(),
                timestamp: new Date().toISOString(),
                content: `Result: ${result}`,
                type: 'output'
              }] : [])
            ]
          }));
        } catch (error) {
          setLogEntries(prev => ({
            ...prev,
            [activeArtifactData.id]: [
              ...(prev[activeArtifactData.id] || []),
              {
                id: crypto.randomUUID(),
                timestamp: new Date().toISOString(),
                content: `Error: ${error.message}`,
                type: 'error'
              }
            ]
          }));
        }
      } else {
        // Handle other languages or call the original onRun function
        onRun(activeArtifactData.id, activeArtifactData.content);
      }
      setShowLogs(true);
    }
  };

  const renderArtifact = (artifact: ArtifactContent) => {
    switch (artifact.type) {
      case 'code':
        return (
          <Code
            key={artifact.id}
            initialContent={artifact.content}
            language={artifact.language || 'javascript'}
            onChange={(value) => onChange(artifact.id, value)}
            onRun={runCode}
          />
        );
      case 'text':
        return <TextPreview key={artifact.id} content={artifact.content} />;
      default:
        return null;
    }
  };

  useEffect(() => {
    if (selectedArtifact) {
      setActiveArtifact(selectedArtifact.id);
    }
  }, [selectedArtifact]);

  return (
    <div className="h-full flex flex-col bg-background rounded-lg border overflow-hidden">
      <ArtifactSidebar
        artifacts={artifacts}
        onArtifactSelect={(artifact) => setActiveArtifact(artifact.id)}
      />
      <div className="p-4 flex-grow flex flex-col">
        <ArtifactTabs
          artifacts={artifacts}
          activeArtifact={activeArtifact}
          setActiveArtifact={setActiveArtifact}
          showPreview={showPreview}
          setShowPreview={setShowPreview}
          showLogs={showLogs}
          setShowLogs={setShowLogs}
          isPreviewable={isPreviewable}
        />
        <ArtifactContent
          showPreview={showPreview}
          showLogs={showLogs}
          activeArtifactData={activeArtifactData}
          logEntries={logEntries}
          artifacts={artifacts}
          activeArtifact={activeArtifact}
          renderArtifact={renderArtifact}
        />
      </div>
      <ArtifactActions
        activeArtifactData={activeArtifactData}
        copied={copied}
        setCopied={setCopied}
        copyToClipboard={copyToClipboard}
        resetContent={() => {/* Implement reset functionality */}}
        runCode={runCode}
        onDelete={onDelete}
        activeArtifact={activeArtifact}
      />
    </div>
  );
}
