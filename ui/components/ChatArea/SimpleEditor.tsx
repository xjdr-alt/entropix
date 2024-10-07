'use client';

import React, { useRef, useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react'
import { Editor, EditorProps, OnMount, loader } from '@monaco-editor/react'
import { useTheme } from 'next-themes'

// Import JSON files
import tomorrow from 'styles/monaco-themes/Tomorrow.json';
import tomorrowNight from 'styles/monaco-themes/Tomorrow-Night.json';
import solarizedLight from 'styles/monaco-themes/Solarized-light.json';
import solarizedDark from 'styles/monaco-themes/Solarized-dark.json';

const editorOptions: EditorProps['options'] = {
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  fontSize: 14,
  fontFamily: "'Fira Code', Menlo, Monaco, 'Courier New', monospace",
  fontLigatures: true,
  folding: false,
  lineNumbers: 'on',
  renderSideBySide: false,
  readOnly: false,
  wordWrap: 'on',
  automaticLayout: true,
  hideCursorInOverviewRuler: true,
  overviewRulerBorder: false,
  overviewRulerLanes: 0,
  renderLineHighlight: 'none',
  scrollbar: {
    vertical: 'hidden',
    horizontal: 'hidden'
  },
  lineDecorationsWidth: 16, // Increase space between border and line numbers
  lineNumbersMinChars: 4, // Increase minimum width of line number column
  glyphMargin: false,
  padding: { top: 16, bottom: 16 }
}

// Updated theme mapping
// const themeMapping: { [key: string]: { light: string; dark: string } } = {
//   'Neutral': { light: tomorrow, dark: tomorrowNight },
//   'Cat': { light: nord, dark: nord },
//   'Slate': { light: solarizedLight, dark: solarizedDark },
//   'Stone': { light: solarizedLight, dark: solarizedDark },
// };

// Theme data mapping
const themeData: { [key: string]: any } = {
  'Tomorrow': tomorrow,
  'Tomorrow-Night': tomorrowNight,
  'Solarized-light': solarizedLight,
  'Solarized-dark': solarizedDark,
};

export const SimpleEditor: React.FC<EditorProps> = ({ value, onChange, className, language = 'markdown' }) => {
  const [internalValue, setInternalValue] = useState(value);
  const editorRef = useRef<any>(null);
  const { resolvedTheme } = useTheme();
  const [monacoTheme, setMonacoTheme] = useState<string>('Solarized-light');

  useEffect(() => {
    if (value !== internalValue) {
      setInternalValue(value);
    }
  }, [value, internalValue]);

  useEffect(() => {
    const selectedTheme = resolvedTheme === 'dark' ? 'Tomorrow-Night' : 'Solarized-light';
    setMonacoTheme(selectedTheme);

    loader.init().then(monaco => {
      monaco.editor.defineTheme(selectedTheme, themeData[selectedTheme]);
    });
  }, [resolvedTheme]);

  const handleEditorDidMount: OnMount = (editor) => {
    editorRef.current = editor;
  };

  const handleEditorChange = (newValue: string | undefined) => {
    const currentPosition = editorRef.current?.getPosition();
    setInternalValue(newValue || '');
    onChange?.(newValue);

    setTimeout(() => {
      if (editorRef.current) {
        editorRef.current.setPosition(currentPosition);
        editorRef.current.focus();
      }
    }, 0);
  };

  return (
    <Editor
      height="calc(100vh - 180px)"
      language={language}
      value={internalValue}
      onChange={handleEditorChange}
      onMount={handleEditorDidMount}
      options={editorOptions}
      loading={<Loader2 className="w-6 h-6 animate-spin" />}
      theme={monacoTheme}
      className={`rounded-lg overflow-hidden ${className}`}
    />
  );
};

export default SimpleEditor;
