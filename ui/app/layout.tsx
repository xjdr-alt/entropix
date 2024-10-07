import React from "react";
import type { Metadata } from "next";
import { Inter, Source_Code_Pro, Fira_Code, Fira_Sans, Manrope } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

const inter = Inter({ subsets: ["latin"] });
// const sourceCodePro = Source_Code_Pro({
//   subsets: ["latin"],
//   weight: ["400", "500", "600", "700"],
//   variable: "--font-source-code-pro",
// });

// const firaCode = Fira_Code({
//   subsets: ["latin"],
//   weight: ["400", "500", "600", "700"],
//   variable: "--font-fira-code",
// });

// const firaSans = Fira_Sans({
//   subsets: ["latin"],
//   weight: ["400", "500", "600", "700"],
//   variable: "--font-fira-sans",
// });

const manrope = Manrope({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-manrope",
});

export const metadata: Metadata = {
  title: "Spire",
  description: "Chat with an AI assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${manrope.variable} flex flex-col h-full`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
