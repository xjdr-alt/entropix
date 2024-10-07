"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import { useTheme } from "next-themes";

const TopNavBar = () => {
  const { theme } = useTheme();
  const [title, setTitle] = useState("Test Chat");
  const [isEditing, setIsEditing] = useState(false);
  const [mounted, setMounted] = useState(false);

  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTitle(e.target.value);
  };

  const handleTitleSubmit = () => {
    setIsEditing(false);
    // You can add logic here to save the title to a backend if needed
  };

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <nav className="text-foreground p-4 flex items-center">
      <div className="font-bold text-xl flex-shrink-0">
        <Image
          src={theme === "dark" ? "/wordmark-dark.svg" : "/wordmark.svg"}
          alt="Company Wordmark"
          width={112}
          height={20}
        />
      </div>
      <div className="flex-grow flex justify-center">
        {isEditing ? (
          <input
            type="text"
            value={title}
            onChange={handleTitleChange}
            onBlur={handleTitleSubmit}
            onKeyPress={(e) => e.key === 'Enter' && handleTitleSubmit()}
            className="text-2xl font-bold bg-transparent border-b-2 border-primary focus:outline-none text-center"
            autoFocus
          />
        ) : (
          <h1
            className="text-2xl font-bold cursor-pointer hover:text-primary transition-colors duration-200"
            onClick={() => setIsEditing(true)}
          >
            {title}
          </h1>
        )}
      </div>
      <div className="flex-shrink-0">
        {/* Placeholder for theme toggles */}
        <div className="w-[112px]"></div>
      </div>
    </nav>
  );
};

export default TopNavBar;