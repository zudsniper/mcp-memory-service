/// <reference types="@cloudflare/workers-types" />

export interface Env {
  DB: D1Database;
  AI: any; // Workers AI binding
}

interface MemoryRow {
  id: string;
  content: string;
  embedding: string;
  tags: string | null;
  metadata: string | null;
  created_at?: number;
}

interface MemoryResult {
  id: string;
  content: string;
  similarity: number;
  tags: any[];
  metadata: any;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const { pathname } = new URL(request.url);

    if (pathname === "/mcp") {
      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();

      const headers = new Headers({
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      });

      ctx.waitUntil(handleMCP(request, env, writer));

      return new Response(readable, { headers });
    }

    if (pathname === "/list_tools") {
      return new Response(JSON.stringify({
        tools: [
          { name: "store_memory", description: "Store new information with optional tags" },
          { name: "retrieve_memory", description: "Find relevant memories based on query" },
          { name: "recall_memory", description: "Retrieve memories using natural language time expressions" },
          { name: "search_by_tag", description: "Search memories by tags" },
          { name: "delete_memory", description: "Delete a specific memory by its hash" },
          { name: "delete_by_tag", description: "Delete all memories with a specific tag" },
          { name: "cleanup_duplicates", description: "Find and remove duplicate entries" },
          { name: "get_embedding", description: "Get raw embedding vector for content" },
          { name: "check_embedding_model", description: "Check if embedding model is loaded and working" },
          { name: "debug_retrieve", description: "Retrieve memories with debug information" },
          { name: "exact_match_retrieve", description: "Retrieve memories using exact content match" },
          { name: "check_database_health", description: "Check database health and get statistics" },
          { name: "recall_by_timeframe", description: "Retrieve memories within a specific timeframe" },
          { name: "delete_by_timeframe", description: "Delete memories within a specific timeframe" },
          { name: "delete_before_date", description: "Delete memories before a specific date" }
        ]
      }, null, 2), {
        headers: { "Content-Type": "application/json" }
      });
    }

    return new Response("Not found", { status: 404 });
  },
};

async function handleDeleteMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const contentHash = args.content_hash;
    if (!contentHash) {
      await sendSSE(writer, { error: "Missing content_hash" });
      return;
    }

    await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(contentHash).run();
    await sendSSE(writer, { result: `Deleted memory with hash ${contentHash}` });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleDeleteByTag(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const tag = args.tag;
    if (!tag) {
      await sendSSE(writer, { error: "Missing tag" });
      return;
    }

    const { results } = await env.DB.prepare("SELECT id, tags FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    let deleteCount = 0;
    for (const row of rows) {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      if (memoryTags.includes(tag)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      }
    }

    await sendSSE(writer, { result: `Deleted ${deleteCount} memories with tag ${tag}` });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleCheckDatabaseHealth(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const { results } = await env.DB.prepare("SELECT COUNT(*) as count FROM memories").all();
    const count = results[0]?.count ?? 0;
    await sendSSE(writer, { status: "healthy", total_memories: count });
  } catch (e) {
    await sendSSE(writer, { status: "error", error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleRecallMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const query = args.query;
    const nResults = args.n_results || 5;
    if (!query) {
      await sendSSE(writer, { error: "Missing query" });
      return;
    }

    const queryEmbedding = await generateEmbedding(query, env);

    const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const scored: MemoryResult[] = [];
    for (const row of rows) {
      const embedding = JSON.parse(row.embedding);
      const sim = cosineSimilarity(queryEmbedding, embedding);
      scored.push({
        id: row.id,
        content: row.content,
        similarity: sim,
        tags: JSON.parse(row.tags ?? "[]"),
        metadata: JSON.parse(row.metadata ?? "{}")
      });
    }

    scored.sort((a, b) => b.similarity - a.similarity);
    const top = scored.slice(0, nResults);

    await sendSSE(writer, { results: top });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleCleanupDuplicates(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const { results } = await env.DB.prepare("SELECT id, content FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const seen = new Map<string, string>();
    let deleteCount = 0;

    for (const row of rows) {
      const hash = row.id;
      if (seen.has(hash)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      } else {
        seen.set(hash, row.content);
      }
    }

    await sendSSE(writer, { result: `Deleted ${deleteCount} duplicate memories` });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleDeleteByTimeframe(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const startDateStr = args.start_date;
    const endDateStr = args.end_date || args.start_date;
    const tag = args.tag;

    if (!startDateStr) {
      await sendSSE(writer, { error: "Missing start_date" });
      return;
    }

    const startTimestamp = new Date(startDateStr).getTime() / 1000;
    const endTimestamp = new Date(endDateStr).getTime() / 1000 + 86399; // end of day

    const { results } = await env.DB.prepare(
      "SELECT id, tags, created_at FROM memories WHERE created_at BETWEEN ? AND ?"
    ).bind(startTimestamp, endTimestamp).all();

    const rows = results as unknown as MemoryRow[];

    let deleteCount = 0;
    for (const row of rows) {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      if (!tag || memoryTags.includes(tag)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      }
    }

    await sendSSE(writer, { deleted: deleteCount });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleGetEmbedding(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const content = args.content;
    if (!content) {
      await sendSSE(writer, { error: "Missing content" });
      return;
    }

    const embedding = await generateEmbedding(content, env);
    await sendSSE(writer, { embedding });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleDeleteBeforeDate(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const beforeDateStr = args.before_date;
    const tag = args.tag;

    if (!beforeDateStr) {
      await sendSSE(writer, { error: "Missing before_date" });
      return;
    }

    const beforeTimestamp = new Date(beforeDateStr).getTime() / 1000;

    const { results } = await env.DB.prepare(
      "SELECT id, tags, created_at FROM memories WHERE created_at < ?"
    ).bind(beforeTimestamp).all();

    const rows = results as unknown as MemoryRow[];

    let deleteCount = 0;
    for (const row of rows) {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      if (!tag || memoryTags.includes(tag)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      }
    }

    await sendSSE(writer, { deleted: deleteCount });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleRecallByTimeframe(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const startDateStr = args.start_date;
    const endDateStr = args.end_date || args.start_date;
    const nResults = args.n_results || 5;

    if (!startDateStr) {
      await sendSSE(writer, { error: "Missing start_date" });
      return;
    }

    const startTimestamp = new Date(startDateStr).getTime() / 1000;
    const endTimestamp = new Date(endDateStr).getTime() / 1000 + 86399; // end of day

    const { results } = await env.DB.prepare(
      "SELECT id, content, tags, metadata, created_at FROM memories WHERE created_at BETWEEN ? AND ? ORDER BY created_at DESC LIMIT ?"
    ).bind(startTimestamp, endTimestamp, nResults).all();

    await sendSSE(writer, { results });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleExactMatchRetrieve(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const content = args.content;
    if (!content) {
      await sendSSE(writer, { error: "Missing content" });
      return;
    }

    const { results } = await env.DB.prepare("SELECT id, content, tags, metadata FROM memories WHERE content = ?").bind(content).all();
    const rows = results as unknown as MemoryRow[];

    await sendSSE(writer, { results: rows });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleCheckEmbeddingModel(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    await generateEmbedding("test", env);
    await sendSSE(writer, { status: "Embedding model is working" });
  } catch (e) {
    await sendSSE(writer, { status: "Embedding model check failed", error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleSearchByTag(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const tags = args.tags;
    if (!tags || !Array.isArray(tags) || tags.length === 0) {
      await sendSSE(writer, { error: "Missing or invalid tags" });
      return;
    }

    const { results } = await env.DB.prepare("SELECT id, content, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const matches = rows.filter(row => {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      return memoryTags.some((t: string) => tags.includes(t));
    });

    await sendSSE(writer, { results: matches });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

interface DebugMemoryResult {
  memory: {
    content: string;
    content_hash: string;
    tags: any[];
    metadata: any;
  };
  debug_info: {
    raw_similarity: number;
    raw_distance: number;
    memory_id: string;
  };
}

async function handleDebugRetrieve(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const query = args.query;
    const nResults = args.n_results || 5;
    const similarityThreshold = args.similarity_threshold || 0.0;
    
    if (!query) {
      await sendSSE(writer, { error: "Missing query" });
      return;
    }
    
    const queryEmbedding = await generateEmbedding(query, env);
    
    const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];
    
    // Calculate similarity and filter by threshold
    const scored: DebugMemoryResult[] = [];
    for (const row of rows) {
      const embedding = JSON.parse(row.embedding);
      const similarity = cosineSimilarity(queryEmbedding, embedding);
      
      // Only include memories with similarity >= threshold
      if (similarity >= similarityThreshold) {
        scored.push({
          memory: {
            content: row.content,
            content_hash: row.id,
            tags: JSON.parse(row.tags ?? "[]"),
            metadata: JSON.parse(row.metadata ?? "{}")
          },
          debug_info: {
            raw_similarity: similarity,
            raw_distance: 1 - similarity,
            memory_id: row.id
          }
        });
      }
    }
    
    // Sort by similarity descending
    scored.sort((a, b) => b.debug_info.raw_similarity - a.debug_info.raw_similarity);
    
    // Limit to n_results
    const top = scored.slice(0, nResults);
    
    if (top.length === 0) {
      await sendSSE(writer, { result: "No matching memories found" });
      return;
    }
    
    // Format the results
    const formattedResults = top.map((result, i) => {
      const memoryInfo = [
        `Memory ${i+1}:`,
        `Content: ${result.memory.content}`,
        `Hash: ${result.memory.content_hash}`,
        `Raw Similarity Score: ${result.debug_info.raw_similarity.toFixed(4)}`,
        `Raw Distance: ${result.debug_info.raw_distance.toFixed(4)}`,
        `Memory ID: ${result.debug_info.memory_id}`
      ];
      
      if (result.memory.tags && result.memory.tags.length > 0) {
        memoryInfo.push(`Tags: ${result.memory.tags.join(', ')}`);
      }
      
      memoryInfo.push("---");
      return memoryInfo.join("\n");
    });
    
    await sendSSE(writer, {
      result: "Found the following memories:\n\n" + formattedResults.join("\n"),
      debug_results: top
    });
  } catch (e) {
    await sendSSE(writer, { error: `Error in debug retrieve: ${e instanceof Error ? e.message : String(e)}` });
  }
}

async function handleStoreMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const content = args.content;
    const metadata = args.metadata || {};
    if (!content) {
      await sendSSE(writer, { error: "Missing content" });
      return;
    }

    const embedding = await generateEmbedding(content, env);

    const encoder = new TextEncoder();
    const data = encoder.encode(content + JSON.stringify(metadata));
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const contentHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

    const tags = metadata.tags || [];
    const memoryType = metadata.type || "";
    const createdAt = Math.floor(Date.now() / 1000);

    await env.DB.prepare(
      `INSERT OR IGNORE INTO memories (id, content, embedding, tags, memory_type, metadata, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      contentHash,
      content,
      JSON.stringify(embedding),
      JSON.stringify(tags),
      memoryType,
      JSON.stringify(metadata),
      createdAt
    ).run();

    await sendSSE(writer, { result: "Memory stored", id: contentHash });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleRetrieveMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const query = args.query;
    const nResults = args.n_results || 5;
    if (!query) {
      await sendSSE(writer, { error: "Missing query" });
      return;
    }

    const queryEmbedding = await generateEmbedding(query, env);

    const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const scored: MemoryResult[] = [];
    for (const row of rows) {
      const embedding = JSON.parse(row.embedding);
      const sim = cosineSimilarity(queryEmbedding, embedding);
      scored.push({
        id: row.id,
        content: row.content,
        similarity: sim,
        tags: JSON.parse(row.tags ?? "[]"),
        metadata: JSON.parse(row.metadata ?? "{}")
      });
    }

    scored.sort((a, b) => b.similarity - a.similarity);
    const top = scored.slice(0, nResults);

    await sendSSE(writer, { results: top });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleMCP(request: Request, env: Env, writer: WritableStreamDefaultWriter) {
  try {
    const body = await request.json() as any;
    const method = body.method;
    const args = body.arguments || {};

    if (method === "store_memory") {
      await handleStoreMemory(args, env, writer);
    } else if (method === "retrieve_memory") {
      await handleRetrieveMemory(args, env, writer);
    } else if (method === "recall_memory") {
      await handleRecallMemory(args, env, writer);
    } else if (method === "recall_by_timeframe") {
      await handleRecallByTimeframe(args, env, writer);
    } else if (method === "exact_match_retrieve") {
      await handleExactMatchRetrieve(args, env, writer);
    } else if (method === "search_by_tag") {
      await handleSearchByTag(args, env, writer);
    } else if (method === "delete_memory") {
      await handleDeleteMemory(args, env, writer);
    } else if (method === "delete_by_tag") {
      await handleDeleteByTag(args, env, writer);
    } else if (method === "cleanup_duplicates") {
      await handleCleanupDuplicates(args, env, writer);
    } else if (method === "get_embedding") {
      await handleGetEmbedding(args, env, writer);
    } else if (method === "check_embedding_model") {
      await handleCheckEmbeddingModel(args, env, writer);
    } else if (method === "debug_retrieve") {
      await handleDebugRetrieve(args, env, writer);
    } else if (method === "check_database_health") {
      await handleCheckDatabaseHealth(args, env, writer);
    } else if (method === "delete_by_timeframe") {
      await handleDeleteByTimeframe(args, env, writer);
    } else if (method === "delete_before_date") {
      await handleDeleteBeforeDate(args, env, writer);
    } else {
      await sendSSE(writer, { error: "Unknown method" });
    }
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  } finally {
    writer.close();
  }
}

async function sendSSE(writer: WritableStreamDefaultWriter, data: any) {
  const payload = `data: ${JSON.stringify(data)}\n\n`;
  await writer.write(new TextEncoder().encode(payload));
}

async function generateEmbedding(text: string, env: Env): Promise<number[]> {
  const model = "@cf/baai/bge-base-en-v1.5";
  const response = await env.AI.run(model, { text });
  if (!response || !response.data || !response.data[0]) {
    throw new Error("Failed to generate embedding");
  }
  return response.data[0];
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}