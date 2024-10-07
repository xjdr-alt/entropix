'use client';

import React from "react";
import TopNavBar from "@/components/TopNavBar";
import ChatArea from "@/components/ChatArea";
import config from "@/config";
import LeftSidebar from "@/components/LeftSidebar";
import RightSidebar from "@/components/RightSidebar";
import Sidebar from "@/components/Sidebar";


export default function Home() {
  return (
    <div className="flex flex-col h-screen w-full">
      <TopNavBar />
      <div className="flex flex-1 overflow-hidden h-screen w-full">
        {/* <LeftSidebar /> */}
        <ChatArea />
        {/* <RightSidebar /> */}
        <Sidebar />
      </div>
    </div>
  );
}