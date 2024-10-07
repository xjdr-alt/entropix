import React from 'react';
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { HandHelping, WandSparkles, BookOpenText, LucideIcon } from "lucide-react";

interface Feature {
  Icon: LucideIcon;
  text: string;
}

const features: Feature[] = [
  { Icon: HandHelping, text: "Need guidance? I'll help navigate tasks using internal resources." },
  { Icon: WandSparkles, text: "I'm a whiz at finding information! I can dig through your knowledge base." },
  { Icon: BookOpenText, text: "I'm always learning! The more you share, the better I can assist you." },
];

const WelcomeMessage: React.FC = () => (
  <div className="flex flex-col items-center justify-center h-full animate-fade-in-up">
    <Avatar className="w-10 h-10 mb-4 border">
      <AvatarImage src="/ant-logo.svg" alt="AI Assistant Avatar" width={40} height={40} />
    </Avatar>
    <h2 className="text-2xl font-semibold mb-8">Here's how I can help</h2>
    <div className="space-y-4 text-sm">
      {features.map(({ Icon, text }, index) => (
        <div key={index} className="flex items-center gap-3">
          <Icon className="text-muted-foreground" />
          <p className="text-muted-foreground">{text}</p>
        </div>
      ))}
    </div>
  </div>
);

export default WelcomeMessage;
