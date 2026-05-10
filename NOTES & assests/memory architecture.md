                    ┌──────────────────────────┐
                    │      User Message        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   MemoryManager Class    │
                    └────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼

┌──────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Short-Term Memory│   │ Long-Term Memory   │   │ Utility Operations │
│ (Conversation)   │   │ (Semantic Facts)   │   │                    │
└────────┬─────────┘   └─────────┬──────────┘   └─────────┬──────────┘
         │                       │                        │
         ▼                       ▼                        ▼

 ┌────────────────┐      ┌──────────────────┐     ┌──────────────────┐
 │ self.history   │      │ semantic_knowledge│    │ clear()          │
 │ List[Dict]     │      │ set()             │    │ reset session    │
 └──────┬─────────┘      └────────┬─────────┘     └──────────────────┘
        │                         │
        ▼                         ▼

┌─────────────────────┐   ┌──────────────────────┐
│ add_message()       │   │ add_semantic_fact()  │
│ Stores chat history │   │ Stores persistent    │
│                     │   │ long-term facts      │
└─────────┬───────────┘   └──────────┬───────────┘
          │                          │
          ▼                          ▼

┌─────────────────────┐   ┌────────────────────────────┐
│ Sliding Window      │   │ Inject facts into          │
│ Trimming            │   │ system prompt              │
│                     │   │                            │
│ Keep only last N    │   │ get_system_prompt_context()│
│ user-assistant msgs │   │                            │
└─────────┬───────────┘   └─────────────┬──────────────┘
          │                             │
          ▼                             ▼

┌──────────────────────┐   ┌─────────────────────────┐
│ get_conversational_  │   │ Dynamic Prompt Context  │
│ window()             │   │                         │
│                      │   │ Example:                │
│ Returns current      │   │ "User prefers concise   │
│ memory window        │   │ answers"                │
└──────────┬───────────┘   └────────────┬────────────┘
           │                            │
           └──────────────┬─────────────┘
                          ▼

              ┌────────────────────────┐
              │  Injected into Agent   │
              │  before LLM call       │
              └────────────────────────┘




--------------------------------------------------------------------------------------

Or like:





                   ┌──────────────────┐
                   │   User Message   │
                   └────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │   MemoryManager    │
                 └────────┬───────────┘
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼

┌──────────────────────┐      ┌────────────────────────┐
│ Short-Term Memory    │      │ Long-Term Memory       │
│ (Conversation Window)│      │ (Semantic Memory)      │
└──────────┬───────────┘      └──────────┬─────────────┘
           │                             │
           ▼                             ▼

 ┌────────────────────┐      ┌────────────────────────┐
 │ self.history       │      │ semantic_knowledge     │
 │ Stores recent msgs │      │ Stores user facts      │
 └─────────┬──────────┘      └──────────┬─────────────┘
           │                             │
           ▼                             ▼

 ┌────────────────────┐      ┌────────────────────────┐
 │ Sliding Window     │      │ Prompt Injection       │
 │ Keeps last N msgs  │      │ Adds facts to system   │
 └─────────┬──────────┘      │ prompt dynamically     │
           │                 └──────────┬─────────────┘
           ▼                            │
     ┌───────────────┐                  │
     │ Current Chat  │                  │
     │ Context       │                  │
     └──────┬────────┘                  │
            └──────────────┬────────────┘
                           ▼

                ┌────────────────────┐
                │   Sent to LLM      │
                │ (Agent + Memory)   │
                └────────────────────┘