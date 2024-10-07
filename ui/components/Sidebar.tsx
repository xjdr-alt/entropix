'use client';

import { useState, useEffect, useRef } from "react";
import { ChevronRight, ChevronLeft, FileText, Plus, Folder, ExternalLink, Moon, Sun, Search, Zap, Star, ChevronDown } from "lucide-react";
import Image from "next/image";
import { useTheme } from "next-themes";
import { themes } from "@/styles/themes";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { motion, AnimatePresence } from "framer-motion";

const themeColors = {
  neutral: "#ffffff",
  cat: "#cba6f7",
  slate: "#64748b",
  stone: "#78716c",
} as const;

type ThemeName = keyof typeof themes;

const ColorCircle = ({ themeName, isSelected }: { themeName: ThemeName; isSelected: boolean }) => (
  <div
    className={`w-4 h-4 rounded-full ${isSelected ? 'ring-2 ring-offset-2 ring-offset-background ring-primary' : ''}`}
    style={{ backgroundColor: themeColors[themeName] }}
  />
);

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [isPinned, setIsPinned] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const { theme, setTheme } = useTheme();
  const [colorTheme, setColorTheme] = useState<ThemeName>("neutral");
  const [mounted, setMounted] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setMounted(true);
    const savedColorTheme = (localStorage.getItem("color-theme") || "neutral") as ThemeName;
    setColorTheme(savedColorTheme);
    applyTheme(savedColorTheme, theme === "dark");
  }, [theme]);

  const applyTheme = (newColorTheme: ThemeName, isDark: boolean) => {
    const root = document.documentElement;
    const themeVariables = isDark ? themes[newColorTheme].dark : themes[newColorTheme].light;

    Object.entries(themeVariables).forEach(([key, value]) => {
      root.style.setProperty(`--${key}`, value as string);
    });
  };

  const handleThemeChange = (newColorTheme: ThemeName) => {
    setColorTheme(newColorTheme);
    localStorage.setItem("color-theme", newColorTheme);
    applyTheme(newColorTheme, theme === "dark");
    window.dispatchEvent(new CustomEvent('themeChange', { detail: newColorTheme }));
  };

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light");
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "/" && !isSearchFocused && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "TEXTAREA") {
        e.preventDefault();
        setIsOpen(true);
        searchInputRef.current?.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isSearchFocused]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isPinned) {
        if (e.clientX <= 10) {
          setIsOpen(true);
        } else if (e.clientX > 320) {
          setIsOpen(false);
        }
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, [isPinned]);

  const togglePin = () => {
    setIsPinned(!isPinned);
    setIsOpen(true);
  };

  const starredChats = [
    "Important Project Discussion",
    "Client Meeting Notes",
  ];

  const recentChats = [
    "Refactoring a React Chat Component",
    "Enhancing a Production-Ready TextE...",
    "Styling UI to Match Claude's Interface",
    "UI Project Assistance",
    "Optimizing System Prompts for AI Ta...",
    "Prompt Conversion to Target Format",
  ];

  const filteredItems = [...starredChats, ...recentChats].filter(item =>
    item.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="relative h-screen">
      <motion.div
        initial={{ x: "-100%" }}
        animate={{ x: isOpen ? 0 : "-100%" }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="fixed top-0 left-0 h-full w-80 bg-black/80 backdrop-blur-md text-gray-300 flex flex-col overflow-hidden"
      >
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <Image
            src="/wordmark-dark.svg"
            alt="Company Wordmark"
            width={112}
            height={20}
          />
          <Button variant="ghost" size="icon" onClick={togglePin} className="text-gray-500 hover:text-gray-300">
            {isPinned ? <ChevronLeft className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            <span className="sr-only">{isPinned ? "Unpin sidebar" : "Pin sidebar"}</span>
          </Button>
        </div>

        <div className="relative p-4">
          <input
            ref={searchInputRef}
            type="text"
            placeholder="(Press '/' to search chats)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => setIsSearchFocused(true)}
            onBlur={() => setIsSearchFocused(false)}
            className="w-full bg-gray-900/50 border border-gray-700 rounded-md py-2 px-4 text-sm text-gray-300 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <Search className="absolute right-7 top-1/2 transform -translate-y-1/2 text-gray-500" size={18} />
        </div>

        <ScrollArea className="flex-grow px-3 py-4">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" className="w-full justify-start text-left mb-4 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white border-none">
                  <Plus className="w-4 h-4 mr-2" />
                  <span className="font-medium">Start new chat</span>
                  <Zap className="w-4 h-4 ml-auto text-yellow-300" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Begin a new AI-powered conversation</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <AnimatePresence>
            {filteredItems.length > 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
                className="space-y-4"
              >
                {starredChats.length > 0 && (
                  <section>
                    <h2 className="text-xs font-semibold text-gray-500 uppercase mb-2">Starred</h2>
                    <ul className="space-y-1">
                      {starredChats.filter(item => item.toLowerCase().includes(searchQuery.toLowerCase())).map((item, index) => (
                        <motion.li
                          key={`starred-${index}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                        >
                          <Button variant="ghost" className="w-full justify-start text-left hover:bg-white/10 group">
                            <Star className="w-4 h-4 mr-2 flex-shrink-0 text-yellow-500" />
                            <span className="truncate">{item}</span>
                            <span className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
                              <ExternalLink className="w-4 h-4" />
                            </span>
                          </Button>
                        </motion.li>
                      ))}
                    </ul>
                  </section>
                )}
                <section>
                  <h2 className="text-xs font-semibold text-gray-500 uppercase mb-2">Recent</h2>
                  <ul className="space-y-1">
                    {recentChats.filter(item => item.toLowerCase().includes(searchQuery.toLowerCase())).map((item, index) => (
                      <motion.li
                        key={`recent-${index}`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                      >
                        <Button variant="ghost" className="w-full justify-start text-left hover:bg-white/10 group">
                          <FileText className="w-4 h-4 mr-2 flex-shrink-0" />
                          <span className="truncate">{item}</span>
                          <span className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
                            <ExternalLink className="w-4 h-4" />
                          </span>
                        </Button>
                      </motion.li>
                    ))}
                  </ul>
                </section>
              </motion.div>
            ) : (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-gray-500 text-center mt-4"
              >
                No matching chats found
              </motion.p>
            )}
          </AnimatePresence>
        </ScrollArea>

        <div className="mt-auto border-t border-gray-800">
          <DropdownMenu open={isDropdownOpen} onOpenChange={setIsDropdownOpen}>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="w-full justify-between p-4 hover:bg-white/10">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center mr-2">
                    <span className="text-xs font-bold text-white">X</span>
                  </div>
                  <span className="text-sm">name@email.com</span>
                </div>
                <ChevronDown className="w-4 h-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-80 bg-black/90 backdrop-blur-md text-gray-300" align="end">
              <div className="p-2">
                <div className="flex items-center space-x-2 mb-2">
                  <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center">
                    <span className="text-lg font-bold text-white">X</span>
                  </div>
                  <div>
                    <p className="font-medium">name@email.com</p>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-purple-400">Personal</span>
                      <span className="text-xs text-gray-400">Pro plan</span>
                    </div>
                  </div>
                </div>
                <div className="bg-white/10 backdrop-blur-sm p-2 rounded-md mb-2">
                  <p className="text-sm text-purple-300">Spire's better with your teammates.</p>
                  <Button variant="link" className="p-0 h-auto text-purple-400 hover:text-purple-300">Add Team Plan</Button>
                </div>
                <DropdownMenuItem>Settings</DropdownMenuItem>
                <DropdownMenuItem>
                  <span>Appearance</span>
                  <ChevronRight className="w-4 h-4 ml-auto" />
                </DropdownMenuItem>
                <DropdownMenuItem>Feature Preview</DropdownMenuItem>
                <DropdownMenuItem>
                  <span>Learn more</span>
                  <ChevronRight className="w-4 h-4 ml-auto" />
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <span>API Console</span>
                  <ExternalLink className="w-4 h-4 ml-auto" />
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <span>Help & Support</span>
                  <ExternalLink className="w-4 h-4 ml-auto" />
                </DropdownMenuItem>
                <DropdownMenuItem>Log Out</DropdownMenuItem>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>

          <div className="p-4 flex items-center justify-end space-x-2 bg-black/60 backdrop-blur-sm">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="w-10 h-10 p-0 rounded-full overflow-hidden">
                  <div className="w-full h-full rounded-full" style={{ backgroundColor: themeColors[colorTheme] }} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {(Object.keys(themes) as ThemeName[]).map((themeName) => (
                  <DropdownMenuItem
                    key={themeName}
                    onClick={() => handleThemeChange(themeName)}
                    className="flex items-center gap-2"
                  >
                    <ColorCircle
                      themeName={themeName}
                      isSelected={colorTheme === themeName}
                    />
                    {themeName.charAt(0).toUpperCase() + themeName.slice(1)}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              className="w-10 h-10 p-0"
            >
              <motion.div
                animate={{ rotate: theme === "dark" ? 180 : 0 }}
                transition={{ type: "spring", stiffness: 200, damping: 10 }}
              >
                {mounted && (theme === "dark" ? (
                  <Moon className="h-5 w-5" />
                ) : (
                  <Sun className="h-5 w-5" />
                ))}
              </motion.div>
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
