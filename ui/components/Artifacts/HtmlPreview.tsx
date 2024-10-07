import React, { useEffect, useRef } from 'react';

interface HtmlPreviewProps {
  content: string;
  language: string;
}

export default function HtmlPreview({ content, language }: HtmlPreviewProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    const transpileAndRender = async () => {
      let transpiledContent = content;

      if (language === 'typescript' || language === 'tsx') {
        try {
          const { code } = await Bun.transform(content, {
            loader: language === 'tsx' ? 'tsx' : 'ts',
            target: 'browser',
          });
          transpiledContent = code;
        } catch (error) {
          console.error('Transpilation error:', error);
          transpiledContent = `Error transpiling ${language}: ${error}`;
        }
      }

      const doc = iframeRef.current?.contentDocument;
      if (doc) {
        doc.open();
        doc.write(transpiledContent);
        doc.close();
      }
    };

    transpileAndRender();
  }, [content, language]);

  return (
    <div className="w-full h-full bg-white dark:bg-gray-800 rounded-lg overflow-hidden">
      <iframe
        ref={iframeRef}
        title="HTML Preview"
        className="w-full h-full border-none"
        sandbox="allow-scripts"
      />
    </div>
  );
}
