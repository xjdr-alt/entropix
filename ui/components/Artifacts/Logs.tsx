'use client';

import React, { useEffect, useRef, useState, useCallback, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChevronRightIcon} from "lucide-react"
import { cn } from "@/library/utils"

interface OutputEntry {
  id: string
  timestamp: string
  content: string
  type: 'log' | 'info' | 'warning' | 'error' | 'success' | 'output'
}

export default function Logs() {
  const outputRef = useRef<HTMLDivElement>(null)
  const [entries, setEntries] = useState<OutputEntry[]>([])
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true)
  const [isMinimized, setIsMinimized] = useState(false)

  const addEntry = useCallback((content: string, type: OutputEntry['type'] = 'output') => {
    setEntries(prevEntries => [...prevEntries, {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      content,
      type,
    }])
  }, [])

  useEffect(() => {
    if (outputRef.current && isScrolledToBottom) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [entries, isScrolledToBottom])

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = event.currentTarget
    setIsScrolledToBottom(scrollHeight - scrollTop === clientHeight)
  }, [])

  // Simulating entries for demonstration
  useEffect(() => {
    const types: OutputEntry['type'][] = ['log', 'info', 'warning', 'error', 'success', 'output']
    const messages = [
      "Initializing quantum neural network...",
      "Syncing with decentralized cloud nodes...",
      "Optimizing AI-driven microservices...",
      "Establishing secure blockchain connection...",
      "Deploying edge computing resources...",
      "Received request: GET /api/v3/quantum-data",
      "Error: Temporal anomaly detected in data stream",
      "Successfully processed 1 million records in 0.1 seconds",
      "> npm run future",
      "$ next quantum-dev",
      "- Quantum server ready on 0.0.0.0:3000, multiverse url: http://localhost:3000"
    ]

    const addRandomEntry = () => {
      const randomType = types[Math.floor(Math.random() * types.length)]
      const randomMessage = messages[Math.floor(Math.random() * messages.length)]
      addEntry(randomMessage, randomType)
    }

    // Add initial entries
    for (let i = 0; i < 5; i++) {
      addRandomEntry()
    }

    const timer = setInterval(addRandomEntry, 2000)

    return () => clearInterval(timer)
  }, [addEntry])

  const typeStyles = {
    error: 'bg-red-500/20 text-red-700 dark:text-red-300 border-red-500/30',
    warning: 'bg-yellow-500/20 text-yellow-700 dark:text-yellow-300 border-yellow-500/30',
    success: 'bg-green-500/20 text-green-700 dark:text-green-300 border-green-500/30',
    info: 'bg-blue-500/20 text-blue-700 dark:text-blue-300 border-blue-500/30',
    log: 'bg-purple-500/20 text-purple-700 dark:text-purple-300 border-purple-500/30',
    output: 'bg-indigo-500/20 text-indigo-700 dark:text-indigo-300 border-indigo-500/30',
  }

  return (
    <motion.div
      className="flex flex-col h-full overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <AnimatePresence>
        {!isMinimized && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="flex-grow"
          >
            <ScrollArea
              className="h-full px-6 py-4"
              ref={outputRef}
              onScroll={handleScroll}
            >
              <div className="font-mono text-sm space-y-3">
                <AnimatePresence>
                  {entries.map((entry) => (
                    <LogEntry key={entry.id} entry={entry} typeStyles={typeStyles} />
                  ))}
                </AnimatePresence>
              </div>
            </ScrollArea>
          </motion.div>
        )}
      </AnimatePresence>
      {!isScrolledToBottom && !isMinimized && (
        <ScrollToBottomButton outputRef={outputRef} />
      )}
    </motion.div>
  )
}

const LogEntry = memo(({ entry, typeStyles }: { entry: OutputEntry, typeStyles: Record<OutputEntry['type'], string> }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.3 }}
    className="flex items-start space-x-4"
  >
    <div className="flex-shrink-0 w-20 text-xs text-gray-500 dark:text-gray-400 mt-1 tabular-nums">
      {new Date(entry.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second: '2-digit'})}
    </div>
    <div className={cn("px-2 py-1 text-xs rounded-full border", typeStyles[entry.type])}>
      {entry.type}
    </div>
    <div className="flex-grow">
      <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-all">
        {entry.type === 'output' && <ChevronRightIcon className="inline-block mr-1 h-4 w-4 text-indigo-600 dark:text-indigo-400" />}
        {entry.content}
      </pre>
    </div>
  </motion.div>
))

const ScrollToBottomButton = memo(({ outputRef }: { outputRef: React.RefObject<HTMLDivElement> }) => (
  <motion.div
    className="absolute bottom-4 right-4"
    initial={{ opacity: 0, scale: 0.8 }}
    animate={{ opacity: 1, scale: 1 }}
    exit={{ opacity: 0, scale: 0.8 }}
    transition={{ duration: 0.2 }}
  >
    <button
      onClick={() => {
        if (outputRef.current) {
          outputRef.current.scrollTop = outputRef.current.scrollHeight
        }
      }}
      className="bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-700 dark:text-indigo-300 rounded-full p-2 transition-colors duration-200 shadow-lg border border-indigo-500/30"
    >
      <ChevronRightIcon className="h-4 w-4 transform rotate-90" />
    </button>
  </motion.div>
))