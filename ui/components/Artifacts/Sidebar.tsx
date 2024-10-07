"use client"

import { useState } from 'react'
import { X, SlidersHorizontal, Code, Image } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTheme } from "next-themes";

interface ArtifactContent {
  id: string;
  type: 'code' | 'image';
  name: string;
  versions: number;
}

interface ArtifactSidebarProps {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  artifacts: ArtifactContent[];
}

export default function ArtifactSidebar({ isOpen, setIsOpen, artifacts }: ArtifactSidebarProps) {
  const { theme } = useTheme();
  const toggleSidebar = () => setIsOpen(!isOpen)

  const sidebarVariants = {
    open: { x: 0 },
    closed: { x: "100%" },
  }

  return (
    <>
      <ToggleButton onClick={toggleSidebar} count={artifacts.length} />

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial="closed"
            animate="open"
            exit="closed"
            variants={sidebarVariants}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed inset-y-0 right-0 w-80 bg-background text-foreground overflow-y-auto shadow-lg mt-16 border-l border-border rounded-l-lg"
          >
            <SidebarContent onClose={toggleSidebar} artifacts={artifacts} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

const ToggleButton = ({ onClick, count }) => (
  <motion.button
    onClick={onClick}
    className="fixed top-4 right-4 z-50 bg-card text-card-foreground p-2 rounded-md flex items-center space-x-1.5 hover:bg-accent transition-colors duration-150 border border-border"
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
    aria-label="Toggle chat controls"
  >
    <SlidersHorizontal size={18} />
    <span className="bg-muted text-muted-foreground text-[10px] rounded-full w-5 h-5 flex items-center justify-center ml-1">
      {count}
    </span>
  </motion.button>
)

const SidebarContent = ({ onClose, artifacts }) => (
  <div className="p-4">
    <SidebarHeader onClose={onClose} />
    <ModelInfo />
    <ArtifactsList artifacts={artifacts} />
    <ContentSection />
  </div>
)

const SidebarHeader = ({ onClose }) => (
  <div className="flex justify-between items-center mb-4">
    <h2 className="text-[15px] font-semibold leading-tight">Chat controls</h2>
    <button onClick={onClose} className="text-muted-foreground hover:text-foreground" aria-label="Close sidebar">
      <X size={18} />
    </button>
  </div>
)

const ModelInfo = () => (
  <div className="mb-6">
    <p className="text-[15px] font-medium leading-tight">Ironwood ATLAS</p>
    <p className="text-[13px] text-gray-400 mt-0.5">
      Most intelligent model{' '}
      <a href="#" className="text-[#7dabf8] hover:underline">
        Learn more
      </a>
    </p>
  </div>
)

const ArtifactsList = ({ artifacts }) => (
  <div className="mb-6">
    <h3 className="text-[15px] font-semibold mb-2 leading-tight">Artifacts</h3>
    <div className="space-y-1">
      {artifacts.map((item) => (
        <ArtifactItem key={item.id} item={item} />
      ))}
    </div>
  </div>
)

const ArtifactItem = ({ item }) => (
  <motion.div
    className="bg-card rounded-md p-2.5 cursor-pointer hover:bg-accent transition-colors duration-150 border border-border"
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
  >
    <div className="flex items-start">
      <span className="text-muted-foreground mr-2.5 mt-0.5 font-mono text-sm">
        {item.type === 'code' ? <Code size={16} /> : <Image size={16} />}
      </span>
      <div>
        <p className="text-[13px] font-medium leading-tight">{item.name}</p>
        <p className="text-[11px] text-muted-foreground mt-0.5">
          Click to open {item.type} • {item.versions} version{item.versions > 1 ? 's' : ''}
        </p>
      </div>
    </div>
  </motion.div>
)

const ContentSection = () => (
  <div>
    <h3 className="text-[15px] font-semibold mb-2 leading-tight">Content</h3>
    <div className="space-y-2">
      {[1, 2].map((item) => (
        <ContentItem key={item} />
      ))}
    </div>
  </div>
)

const ContentItem = () => (
  <motion.div
    className="bg-card rounded-md p-2 flex items-center cursor-pointer hover:bg-accent transition-colors duration-150 border border-border"
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
  >
    <div className="w-14 h-14 bg-muted rounded-md mr-3 flex items-center justify-center">
      <span className="text-muted-foreground text-2xl">□</span>
    </div>
    <span className="text-[13px]">image.png</span>
  </motion.div>
)