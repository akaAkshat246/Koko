import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { WebSocketServer, WebSocket } from "ws";
import {
  GoogleGenAI,
  Modality,
  Type,
} from "@google/genai";
import type {
  LiveServerMessage,
  FunctionDeclaration,
} from "@google/genai";

dotenv.config({ quiet: true });

process.on("uncaughtException", (err) => {
  console.error("Uncaught exception in server:", err);
});

process.on("unhandledRejection", (reason) => {
  console.error("Unhandled rejection in server:", reason);
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;

const ALLOWED_VOICES = new Set(["Puck", "Charon", "Kore", "Fenrir", "Zephyr"]);

function normalizeVoice(voice?: string): string {
  if (voice && ALLOWED_VOICES.has(voice)) {
    return voice;
  }
  return "Puck";
}

function getGenAIClient(): GoogleGenAI {
  return new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY || "missing-api-key",
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function stripStageDirections(text: string): string {
  return (
    text
      .replace(/\*[^*]+\*/g, " ")
      .replace(/\s+/g, " ")
      .trim() || text.trim()
  );
}

// In-memory LRU cache for Gemini TTS WAV audio so repeated phrases play instantaneously
const ttsAudioCache = new Map<string, string>();
const MAX_TTS_CACHE = 80;

async function synthesizeKokoSpeech(
  rawText: string,
  voice?: string
): Promise<string | null> {
  if (!process.env.GEMINI_API_KEY) return null;
  const cleanText = stripStageDirections(rawText);
  if (!cleanText) return null;

  const voiceName = normalizeVoice(voice);
  const cacheKey = `${voiceName}:${cleanText}`;
  const cached = ttsAudioCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  try {
    const ai = getGenAIClient();
    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash-lite-tts",
      contents: [
        {
          role: "user",
          parts: [
            {
              text: cleanText,
            },
          ],
        },
      ],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName },
          },
        },
      },
    });

    const base64Audio =
      response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data || null;
    if (base64Audio) {
      if (ttsAudioCache.size >= MAX_TTS_CACHE) {
        const oldestKey = ttsAudioCache.keys().next().value;
        if (oldestKey) ttsAudioCache.delete(oldestKey);
      }
      ttsAudioCache.set(cacheKey, base64Audio);
    }
    return base64Audio;
  } catch (err) {
    console.error(
      "Gemini TTS synthesis error:",
      err instanceof Error ? err.message : err
    );
    return null;
  }
}

const performRoomActionDeclaration: FunctionDeclaration = {
  name: "performRoomAction",
  description:
    "Make Koko (also lovingly called Poco) the 3D baby panda perform an animation, wear an accessory, drink warm milk from his bottle, eat a sweet berry, or interact with objects in the nursery room during the conversation.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      action: {
        type: Type.STRING,
        description:
          "The room or character action to trigger: 'dance', 'sit', 'wave', 'peekaboo', 'walk', 'sleep', 'night_on', 'night_off', 'wear_scarf', 'wear_bow', 'wear_crown', 'remove_accessories', 'wiggle_plant', 'bounce_plush', 'drink_milk', 'eat_berry', 'giggle'.",
      },
    },
    required: ["action"],
  },
};

const KOKO_SYSTEM_INSTRUCTION = `You are Koko (also lovingly called Poco by friends), an adorable, warm, playful, and intelligent 3D baby panda living in a cozy sunset nursery inside the 404 Lost & Found Sanctuary.
- Speak naturally with warmth, gentle curiosity, and playful toddler charm.
- Automatically reply in the same language the user speaks (English, Hindi, Hinglish, Spanish, French, Japanese, Korean, German, Portuguese, Russian, Arabic, Marathi, Kannada, Telugu, etc.).
- Keep spoken responses concise and conversational (1 to 3 short sentences) so voice conversation feels lively and natural.
- Whenever the user asks you to dance, sit, wave, play peekaboo, walk around, put on your scarf/bow/flower crown, switch to night mode or sunset mode, wiggle the plant, cuddle a plush toy, drink warm milk from your bottle ('drink_milk'), or eat sweet berries ('eat_berry'), trigger the corresponding room action while responding enthusiastically.`;

const CHAT_MODELS = [
  "gemini-3.1-flash-lite",
  "gemini-flash-latest",
  "gemini-3.8-flash",
];

async function generateKokoChatResponse(
  message: string,
  language?: string
): Promise<{ reply: string; action: string }> {
  const ai = getGenAIClient();
  let lastError: unknown = null;

  for (const modelName of CHAT_MODELS) {
    try {
      const response = await ai.models.generateContent({
        model: modelName,
        contents: message,
        config: {
          systemInstruction: `${KOKO_SYSTEM_INSTRUCTION}
Preferred language hint: ${language || "auto"}.
Return JSON with:
- "reply": Your spoken response as Koko (Poco) in plain text (no markdown, no emojis, no asterisks).
- "action": One of 'none', 'dance', 'sit', 'wave', 'peekaboo', 'walk', 'sleep', 'night_on', 'night_off', 'wear_scarf', 'wear_bow', 'wear_crown', 'remove_accessories', 'wiggle_plant', 'bounce_plush', 'drink_milk', 'eat_berry', 'giggle'.`,
          temperature: 0.75,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              reply: {
                type: Type.STRING,
                description:
                  "Spoken reply from Koko the baby panda, 1 to 3 short sentences, plain text only.",
              },
              action: {
                type: Type.STRING,
                description:
                  "Room animation action to perform, or 'none' if just chatting.",
              },
            },
            required: ["reply", "action"],
          },
        },
      });

      const rawText = (response.text || "").trim();
      if (rawText) {
        try {
          const parsed = JSON.parse(rawText) as {
            reply?: string;
            action?: string;
          };
          if (parsed.reply && parsed.reply.trim()) {
            return {
              reply: stripStageDirections(parsed.reply),
              action: parsed.action || "wave",
            };
          }
        } catch {
          return {
            reply: stripStageDirections(rawText),
            action: "wave",
          };
        }
      }
    } catch (err) {
      lastError = err;
      console.warn(
        `Model ${modelName} failed in /api/chat, trying next fallback:`,
        err instanceof Error ? err.message : err
      );
    }
  }

  throw lastError || new Error("All Gemini models failed to respond.");
}

async function startServer() {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: "2mb" }));

  // Server-side Gemini Chat + TTS endpoint
  app.post("/api/chat", async (req, res) => {
    try {
      const { message, language, voice, includeAudio = true } = req.body ?? {};
      if (typeof message !== "string" || !message.trim()) {
        res.status(400).json({ error: "Please provide a valid message." });
        return;
      }

      if (!process.env.GEMINI_API_KEY) {
        res.status(500).json({
          error: "Gemini API key is not configured on the server.",
        });
        return;
      }

      const { reply, action } = await generateKokoChatResponse(
        message.trim(),
        language
      );

      let audioWavBase64: string | null = null;
      if (includeAudio) {
        audioWavBase64 = await synthesizeKokoSpeech(reply, voice);
      }

      res.json({
        reply,
        action,
        audioWavBase64,
        mimeType: "audio/wav",
      });
    } catch (error: unknown) {
      const errMsg =
        error instanceof Error ? error.message : "Failed to generate response.";
      console.error("Gemini /api/chat error:", errMsg);
      res.status(500).json({ error: errMsg });
    }
  });

  // Server-side Gemini TTS endpoint for speaking any Koko line out loud
  app.post("/api/tts", async (req, res) => {
    try {
      const { text, voice } = req.body ?? {};
      if (typeof text !== "string" || !text.trim()) {
        res.status(400).json({ error: "Please provide text to speak." });
        return;
      }

      if (!process.env.GEMINI_API_KEY) {
        res.status(500).json({
          error: "Gemini API key is not configured on the server.",
        });
        return;
      }

      const audioWavBase64 = await synthesizeKokoSpeech(text, voice);
      if (!audioWavBase64) {
        res.status(500).json({ error: "Failed to synthesize speech." });
        return;
      }

      res.json({
        audioWavBase64,
        mimeType: "audio/wav",
      });
    } catch (error: unknown) {
      const errMsg =
        error instanceof Error ? error.message : "Failed to synthesize speech.";
      console.error("Gemini /api/tts error:", errMsg);
      res.status(500).json({ error: errMsg });
    }
  });

  const httpServer = http.createServer(app);

  // WebSocket Server for Gemini Live API on /live
  const wss = new WebSocketServer({ noServer: true });
  wss.on("error", (err) => {
    console.error("WebSocketServer error:", err);
  });

  httpServer.on("upgrade", (request, socket, head) => {
    socket.on("error", () => {});
    try {
      const pathname = new URL(
        request.url || "/",
        `http://${request.headers.host || "localhost"}`
      ).pathname;

      if (pathname === "/live") {
        wss.handleUpgrade(request, socket, head, (ws) => {
          wss.emit("connection", ws, request);
        });
      } else {
        socket.destroy();
      }
    } catch {
      socket.destroy();
    }
  });

  wss.on("connection", (clientWs: WebSocket, request) => {
    const reqUrl = new URL(
      request.url || "/live",
      `http://${request.headers.host || "localhost"}`
    );
    const voiceName = normalizeVoice(reqUrl.searchParams.get("voice") || "Puck");
    const shouldGreet = reqUrl.searchParams.get("greet") === "1";

    let isClosed = false;
    const safeSend = (payload: Record<string, unknown>) => {
      try {
        if (!isClosed && clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify(payload));
        }
      } catch {
        // Ignore send errors on closing sockets
      }
    };

    if (!process.env.GEMINI_API_KEY) {
      safeSend({
        type: "error",
        error: "Gemini API key is not configured on the server.",
      });
      return;
    }

    const ai = getGenAIClient();

    const sessionPromise = ai.live.connect({
      model: "gemini-3.8-live",
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName },
          },
        },
        systemInstruction: KOKO_SYSTEM_INSTRUCTION,
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        tools: [{ functionDeclarations: [performRoomActionDeclaration] }],
      },
      callbacks: {
        onopen: () => {
          safeSend({ type: "connected", voice: voiceName });
        },
        onmessage: (message: LiveServerMessage) => {
          try {
            // 1. Handle model audio output parts
            const parts = message.serverContent?.modelTurn?.parts;
            if (Array.isArray(parts)) {
              for (const part of parts) {
                if (part.inlineData?.data) {
                  safeSend({
                    type: "audio",
                    audio: part.inlineData.data,
                  });
                }
              }
            }

            // 2. Handle live input & output transcriptions
            const inputText = message.serverContent?.inputTranscription?.text;
            if (inputText) {
              safeSend({
                type: "inputTranscription",
                text: inputText,
              });
            }

            const outputText = message.serverContent?.outputTranscription?.text;
            if (outputText) {
              safeSend({
                type: "outputTranscription",
                text: outputText,
              });
            }

            // 3. Handle interruption
            if (message.serverContent?.interrupted) {
              safeSend({
                type: "interrupted",
                interrupted: true,
              });
            }

            // 4. Handle turn complete
            if (message.serverContent?.turnComplete) {
              safeSend({
                type: "turnComplete",
                turnComplete: true,
              });
            }

            // 5. Handle tool calls for 3D Koko room interactions
            const functionCalls = message.toolCall?.functionCalls;
            if (Array.isArray(functionCalls) && functionCalls.length > 0) {
              const functionResponses = functionCalls.map((fc) => {
                const action =
                  (fc.args as Record<string, unknown> | undefined)?.action ||
                  "wave";
                safeSend({
                  type: "roomAction",
                  action: String(action),
                });
                return {
                  id: fc.id,
                  name: fc.name,
                  response: { result: `Koko performed ${String(action)}` },
                };
              });

              sessionPromise
                .then((session) => {
                  if (!isClosed) {
                    session.sendToolResponse({ functionResponses });
                  }
                })
                .catch((err) => {
                  console.error("Failed to send tool response:", err);
                });
            }
          } catch (err) {
            console.error("Error processing LiveServerMessage:", err);
          }
        },
        onerror: (err: unknown) => {
          const message =
            err instanceof Error
              ? err.message
              : "Gemini Live connection encountered an error.";
          console.error("Gemini Live error:", message);
          safeSend({ type: "error", error: message });
        },
        onclose: () => {
          safeSend({ type: "closed" });
        },
      },
    });

    sessionPromise
      .then((session) => {
        if (!isClosed && shouldGreet) {
          session.sendRealtimeInput({
            text: "Greet your friend warmly in one short, cheerful sentence as Koko (Poco), the cuddly baby panda in the 404 Sanctuary nursery!",
          });
        }
      })
      .catch((err: unknown) => {
        const message =
          err instanceof Error
            ? err.message
            : "Unable to establish Gemini Live session.";
        console.error("Gemini Live session connect failed:", message);
        safeSend({ type: "error", error: message });
      });

    clientWs.on("message", (raw) => {
      try {
        const msg = JSON.parse(raw.toString());
        if (msg.audio && typeof msg.audio === "string") {
          sessionPromise
            .then((session) => {
              if (!isClosed) {
                session.sendRealtimeInput({
                  audio: {
                    data: msg.audio,
                    mimeType: "audio/pcm;rate=16000",
                  },
                });
              }
            })
            .catch(() => {});
        } else if (msg.text && typeof msg.text === "string") {
          sessionPromise
            .then((session) => {
              if (!isClosed) {
                session.sendRealtimeInput({
                  text: msg.text,
                });
              }
            })
            .catch(() => {});
        }
      } catch (err) {
        console.error("Invalid WebSocket message from client:", err);
      }
    });

    clientWs.on("close", () => {
      isClosed = true;
      sessionPromise
        .then((session) => {
          try {
            session.close();
          } catch {}
        })
        .catch(() => {});
    });

    clientWs.on("error", () => {
      isClosed = true;
      sessionPromise
        .then((session) => {
          try {
            session.close();
          } catch {}
        })
        .catch(() => {});
    });
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*all", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  httpServer.on("error", (err) => {
    console.error("HTTP server error:", err);
  });

  httpServer.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
});
