# TODO

I somewhat hastily removed a bunch of backend code to get this pushed, so the current repo is in kind of rough shape. It runs, but its all stubbed out with mock data. We more or less need to make everything all over again.

This is the initial TODO list but we will add to it as we think of things. 

## REPO
- Clean up repo. I am not a front end developer and it shows. Update the ui folder to best practices while still using bun, shadcn, next and tailwind
- Storybook, jest, etc? This is probably too much but a subset might be useful
- automation, piplines, dockerfiles, etc

## UI
- Markdown rendering in the MessageArea. Make sure we are using rehype and remark properly. Make sure we have the proper code theme based on the selected app theme
  - latex rendering
  - image rendering
- Fix HTML / React Artifact rendering. Had to rip out the old code, so we need to mostly make this from scratch
- Wire up right sidebar to properly handle the artifacts  
- For now hook up pyodide or something like https://github.com/cohere-ai/cohere-terrarium to run python code to start. I will port over the real code-interpreter at some point in the future
- Hook up play button to python interpreter / HTML Viewer
- Hook up CoT parsing and wire it up to the logs tab in the right sidebar OR repurpose the LeftSidebar for CoT viewing
- Hook up Sidebar to either LocalDB, IndexDB or set up docker containers to run postgres (this probably means Drizzle, ughhhh....) to preserve chat history
- Hook up Sidebar search
- Port over or make new keyboard shortcuts
- Create new conversation forking logic and UI. Old forking logic and UI were removed (modal editor was kept) but this is by far one of the most important things to get right
- Visualize entropy / varent via shadcn charts / color the text on the screen
- add shadcn dashboard-03 (the playground) back in for not Claude.ai style conversations

## Editor
- I'm pretty sure i'm not doing Monaco as well as it can be done. Plugins, themes, etc
- do something like https://github.com/Porter97/monaco-copilot-demo with base for completion
- make it work like OAI canvas where you can ask for edits at point
- Make sure Modal Editor and Artifact Code Editor both work but do not rely on eachother, cause ModalEditor needs to be simple

## Backend
- Make a simple SSE client / server to hook up to Entropix generate loop
- Create tool parser for:
  - Brave
  - iPython
  - Image
 
