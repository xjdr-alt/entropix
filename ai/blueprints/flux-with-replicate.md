# Flux with Replicate Documentation

Use this guide to setup generating images with Flux and Replicate.

Write the complete code for every step. Do not get lazy. Write everything that is needed.

Your goal is to completely finish the feature.

## Helpful Links

- [Replicate](https://replicate.com)
- [Flux Schnell](https://replicate.com/black-forest-labs/flux-schnell?input=nodejs)

## Required Environment Variables

Make sure the user has the following environment variables set:

- REPLICATE_API_TOKEN=

## Install Replicate

Make sure the user has the Replicate package installed:

```bash
npm install replicate
```

## Setup Steps

### Create a Replicate Client

This file should go in `/lib/replicate.ts`

```ts
import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});
```

### Create a Server Action

This file should go in `/actions/replicate-actions.ts`

```ts
"use server";

import replicate from "@/lib/replicate";

export async function generateFluxImage(prompt: string) {
const input = {
prompt: prompt,
num_outputs: 1,
aspect_ratio: "1:1",
output_format: "webp",
output_quality: 80
};

const output = await http://replicate.run("black-forest-labs/flux-schnell", { input });
return output;
}
```

### Build the Frontend

This file should go in `/app/flux/page.tsx`.

- Create a form that takes a prompt
- Create a button that calls the server action
- Have a nice ui for when the image is blank or loading
- Display the image that is returned
- Have a button to generate a new image
- Have a button to download the image
