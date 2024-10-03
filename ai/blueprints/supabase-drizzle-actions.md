# Backend Setup Instructions

Use this guide to setup the backend for this project.

It uses Supabase, Drizzle ORM, and Server Actions.

Write the complete code for every step. Do not get lazy. Write everything that is needed.

Your goal is to completely finish the backend setup.

## Helpful Links

If the user gets stuck, refer them to the following links:

- [Supabase Docs](https://supabase.com)
- [Drizzle Docs](https://orm.drizzle.team/docs/overview)
- [Drizzle with Supabase Quickstart](https://orm.drizzle.team/learn/tutorials/drizzle-with-supabase)

## Install Libraries

Make sure the user knows to install the following libraries:

```bash
npm i drizzle-orm dotenv postgres
npm i -D drizzle-kit
```

## Setup Steps

- [ ] Create a `/db` folder in the root of the project

- [ ] Create a `/types` folder in the root of the project

- [ ] Add a `drizzle.config.ts` file to the root of the project with the following code:

```ts
import { config } from "dotenv";
import { defineConfig } from "drizzle-kit";

config({ path: ".env.local" });

export default defineConfig({
  schema: "./db/schema/index.ts",
  out: "./db/migrations",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL!,
  },
});
```

- [ ] Add a file called `db.ts` to the `/db` folder with the following code:

```ts
import { config } from "dotenv";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import { exampleTable } from "./schema";

config({ path: ".env.local" });

const schema = {
  exampleTable,
};

const client = postgres(process.env.DATABASE_URL!);

export const db = drizzle(client, { schema });
```

- [ ] Create 2 folders in the `/db` folder:

- `/schema`
- Add a file called `index.ts` to the `/schema` folder
- `/queries`

- [ ] Create an example table in the `/schema` folder called `example-schema.ts` with the following code:

```ts
import { integer, pgTable, text, timestamp, uuid } from "drizzle-orm/pg-core";

export const exampleTable = pgTable("example", {
  id: uuid("id").defaultRandom().primaryKey(),
  name: text("name").notNull(),
  age: integer("age").notNull(),
  email: text("email").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at")
    .notNull()
    .defaultNow()
    .$onUpdate(() => new Date()),
});

export type InsertExample = typeof exampleTable.$inferInsert;
export type SelectExample = typeof exampleTable.$inferSelect;
```

- [ ] Export the example table in the `/schema/index.ts` file like so:

```ts
export * from "./example-schema";
```

- [ ] Create a new file called `example-queries.ts` in the `/queries` folder with the following code:

```ts
"use server";

import { eq } from "drizzle-orm";
import { db } from "../db";
import { InsertExample, SelectExample } from "../schema/example-schema";
import { exampleTable } from "./../schema/example-schema";

export const createExample = async (data: InsertExample) => {
try {
const [newExample] = await db.insert(exampleTable).values(data).returning();
return newExample;
} catch (error) {
console.error("Error creating example:", error);
throw new Error("Failed to create example");
}
};

export const getExampleById = async (id: string) => {
try {
const example = await db.query.exampleTable.findFirst({
where: eq(http://exampleTable.id, id)
});
if (!example) {
throw new Error("Example not found");
}
return example;
} catch (error) {
console.error("Error getting example by ID:", error);
throw new Error("Failed to get example");
}
};

export const getAllExamples = async (): Promise<SelectExample[]> => {
return db.query.exampleTable.findMany();
};

export const updateExample = async (id: string, data: Partial<InsertExample>) => {
try {
const [updatedExample] = await db.update(exampleTable).set(data).where(eq(http://exampleTable.id, id)).returning();
return updatedExample;
} catch (error) {
console.error("Error updating example:", error);
throw new Error("Failed to update example");
}
};

export const deleteExample = async (id: string) => {
try {
await db.delete(exampleTable).where(eq(http://exampleTable.id, id));
} catch (error) {
console.error("Error deleting example:", error);
throw new Error("Failed to delete example");
}
};
```

- [ ] In `package.json`, add the following scripts:

```json
"scripts": {
"db:generate": "npx drizzle-kit generate",
"db:migrate": "npx drizzle-kit migrate"
}
```

- [ ] Run the following command to generate the tables:

```bash
npm run db:generate
```

- [ ] Run the following command to migrate the tables:

```bash
npm run db:migrate
```

- [ ] Create a folder called `/actions` in the root of the project for server actions

- [ ] Create folder called `/types` in the root of the project for shared types

- [ ] Create a file called `action-types.ts` in the `/types/actions` folder for server action types with the following code:

- [ ] Create file called `/types/index.ts` and export all the types from the `/types` folder like so:

```ts
export * from "./action-types";
```

- [ ] Create a file called `example-actions.ts` in the `/actions` folder for the example table's actions:

```ts
"use server";

import {
  createExample,
  deleteExample,
  getAllExamples,
  getExampleById,
  updateExample,
} from "@/db/queries/example-queries";
import { InsertExample } from "@/db/schema/example-schema";
import { ActionState } from "@/types";
import { revalidatePath } from "next/cache";

export async function createExampleAction(
  data: InsertExample
): Promise<ActionState> {
  try {
    const newExample = await createExample(data);
    revalidatePath("/examples");
    return {
      status: "success",
      message: "Example created successfully",
      data: newExample,
    };
  } catch (error) {
    return { status: "error", message: "Failed to create example" };
  }
}

export async function getExampleByIdAction(id: string): Promise<ActionState> {
  try {
    const example = await getExampleById(id);
    return {
      status: "success",
      message: "Example retrieved successfully",
      data: example,
    };
  } catch (error) {
    return { status: "error", message: "Failed to get example" };
  }
}

export async function getAllExamplesAction(): Promise<ActionState> {
  try {
    const examples = await getAllExamples();
    return {
      status: "success",
      message: "Examples retrieved successfully",
      data: examples,
    };
  } catch (error) {
    return { status: "error", message: "Failed to get examples" };
  }
}

export async function updateExampleAction(
  id: string,
  data: Partial<InsertExample>
): Promise<ActionState> {
  try {
    const updatedExample = await updateExample(id, data);
    revalidatePath("/examples");
    return {
      status: "success",
      message: "Example updated successfully",
      data: updatedExample,
    };
  } catch (error) {
    return { status: "error", message: "Failed to update example" };
  }
}

export async function deleteExampleAction(id: string): Promise<ActionState> {
  try {
    await deleteExample(id);
    revalidatePath("/examples");
    return { status: "success", message: "Example deleted successfully" };
  } catch (error) {
    return { status: "error", message: "Failed to delete example" };
  }
}
```

```ts
export type ActionState = {
  status: "success" | "error";
  message: string;
  data?: any;
};
```

- [ ] Implement the server actions in the `/app/page.tsx` file to allow for manual testing.

- [ ] The backend is now setup.
