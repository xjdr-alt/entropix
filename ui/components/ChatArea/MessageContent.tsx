import React, { useState, useEffect, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';
import { useTheme } from 'next-themes';
import { MessageContentState, ParsedResponse } from '@/types/chat';

// Import only the languages you need
import js from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import ruby from 'react-syntax-highlighter/dist/esm/languages/prism/ruby';

SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('typescript', typescript);
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('ruby', ruby);

interface MessageContentProps {
  content: string;
  role: string;
}

const MessageContent: React.FC<MessageContentProps> = ({ content, role }) => {
  const { theme } = useTheme();
  const [state, setState] = useState<MessageContentState>({
    thinking: true,
    parsed: { response: '' },
    error: false
  });

  useEffect(() => {
    if (role !== "assistant" || !content) {
      setState({ thinking: false, parsed: { response: '' }, error: false });
      return;
    }

    const timer = setTimeout(() => setState(s => ({ ...s, thinking: false, error: true })), 30000);

    try {
      const result = JSON.parse(content) as ParsedResponse;
      console.log("🔍 Parsed Result:", result);

      if (result.response && result.response.length > 0 && result.response !== "...") {
        setState({ thinking: false, parsed: result, error: false });
        clearTimeout(timer);
      }
    } catch (error) {
      console.error("Error parsing JSON:", error);
      setState({ thinking: false, parsed: { response: '' }, error: true });
    }

    return () => clearTimeout(timer);
  }, [content, role]);

  const memoizedContent = useMemo(() => {
    if (state.parsed.response) {
      return state.parsed.response;
    }
    return content;
  }, [state.parsed.response, content]);

  const syntaxTheme = theme === 'dark' ? oneDark : oneLight;

  // Update this line to determine text color based on theme and role
  const textColor = (theme === 'dark') !== (role === 'user') ? 'rgb(255, 255, 255)' : 'rgb(0, 0, 0)';

  if (state.thinking && role === "assistant") {
    return (
      <div className="flex items-center justify-start space-x-2 h-8">
        <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
        <div className="w-2 h-2 bg-primary rounded-full animate-pulse delay-75" />
        <div className="w-2 h-2 bg-primary rounded-full animate-pulse delay-150" />
      </div>
    );
  }

  if (state.error && !state.parsed.response) {
    return <div className="text-destructive">Something went wrong. Please try again.</div>;
  }

  return (
    <div className="prose dark:prose-invert max-w-none" style={{ color: textColor }}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeSanitize, rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={syntaxTheme}
                language={match[1]}
                PreTag="div"
                className="rounded-md my-4"
                showLineNumbers={true}
                wrapLines={true}
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className="px-1 py-0.5 rounded-sm bg-muted" style={{ color: textColor }} {...props}>
                {children}
              </code>
            );
          },
          img({ node, ...props }) {
            return (
              <img
                {...props}
                className="max-w-full h-auto rounded-lg shadow-md transition-shadow duration-300 hover:shadow-lg"
                loading="lazy"
              />
            );
          },
          table({ node, ...props }) {
            return (
              <div className="overflow-x-auto my-4">
                <table className="min-w-full divide-y divide-border border border-border" {...props} />
              </div>
            );
          },
          th({ node, ...props }) {
            return <th className="px-3 py-2 bg-muted text-left text-xs font-medium text-muted-foreground uppercase tracking-wider" {...props} />;
          },
          td({ node, ...props }) {
            return <td className="px-3 py-2 whitespace-nowrap text-sm text-muted-foreground border-t border-border" {...props} />;
          },
          blockquote({ node, ...props }) {
            return (
              <blockquote
                className="border-l-4 border-primary pl-4 py-2 italic text-muted-foreground bg-muted rounded-r-md my-4"
                {...props}
              />
            );
          },
          a({ node, ...props }) {
            return (
              <a
                className="text-primary hover:text-primary/80 transition-colors duration-200"
                target="_blank"
                rel="noopener noreferrer"
                {...props}
              />
            );
          },
          h1({ node, ...props }) {
            return <h1 style={{ color: textColor }} className="text-2xl font-bold mt-6 mb-4" {...props} />;
          },
          h2({ node, ...props }) {
            return <h2 style={{ color: textColor }} className="text-xl font-semibold mt-5 mb-3" {...props} />;
          },
          h3({ node, ...props }) {
            return <h3 style={{ color: textColor }} className="text-lg font-medium mt-4 mb-2" {...props} />;
          },
          h4({ node, ...props }) {
            return <h4 style={{ color: textColor }} className="text-base font-medium mt-3 mb-2" {...props} />;
          },
          h5({ node, ...props }) {
            return <h5 style={{ color: textColor }} className="text-sm font-medium mt-3 mb-1" {...props} />;
          },
          h6({ node, ...props }) {
            return <h6 style={{ color: textColor }} className="text-sm font-medium mt-3 mb-1" {...props} />;
          },
          p({ node, ...props }) {
            return <p style={{ color: textColor }} className="my-3 leading-relaxed" {...props} />;
          },
          ul({ node, ...props }) {
            return <ul className="list-disc list-inside my-3 pl-4" {...props} />;
          },
          ol({ node, ...props }) {
            return <ol className="list-decimal list-inside my-3 pl-4" {...props} />;
          },
          li({ node, ...props }) {
            return <li className="my-1" {...props} />;
          },
          hr({ node, ...props }) {
            return <hr className="my-6 border-gray-200 dark:border-gray-700" {...props} />;
          }
        }}
      >
        {memoizedContent}
      </ReactMarkdown>
    </div>
  );
};

export default MessageContent;