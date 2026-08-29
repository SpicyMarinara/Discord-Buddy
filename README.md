# Discord Buddy

## Discord AI Companion Bot
Open source, free, with prompts written by yours truly. They're like real people, but better, because they actually talk to you!

![image/jpeg](https://i.imgur.com/MZGwi7c.jpeg "Art by https://x.com/sevequiem")

## Installation

### Bot

1. Go to Discord Developer App https://discord.com/developers/applications
2. Create a New Application for your bot.
3. Go to the General Information tab.
4. Name your bot and give it a profile picture and description.

![image/gif](https://i.imgur.com/bzIBYg1.gif)

5. Go to the Installation tab.
6. Add "bot" in Guild Install Scope and "Administrator" in Permissions.
7. Copy the Install Link.

![image/gif](https://i.imgur.com/QJi1dJW.gif)

8. Invite the bot to your selected server(s) with the link.
   
![image/gif](https://i.imgur.com/4KA4cPZ.gif)

9. Return to the Installation tab in Discord Developer App and change Install Link to None, to prevent others from inviting your bot to their servers (unless you don't mind).

![image/gif](https://i.imgur.com/iI023XH.gif)

10. Go to the Bot tab.
11. Enable Presence Intent, Server Members Intent, and Message Content Intent toggles.

![image/gif](https://i.imgur.com/Vo6ACPb.gif)

12. Scroll to the top of the page and click the Reset Token button.
13. Confirm, complete the authentication (if you have it), and Copy your generated token.
14. Save the token for later (somewhere private). DO NOT SHARE IT WITH ANYONE.

![image/gif](https://i.imgur.com/bEoqo6t.gif)

### Setup

1. Get Miniconda3 from here: https://www.anaconda.com/download/success
2. Download all the Git files and unzip them.
3. Go to the downloaded folder.
4. Run one of the two setup files, depending on your system:

- For Windows (double click it):
```
setup_windows.bat
```

- For MacOS/Linux (in console):
```
cd Discord_Buddy
sh ./setup_linux_macos.sh
```

5. Set your Discord Token and API Keys in .env (if it's hidden on Mac, press `shift + command + .` to display hidden files).

![image/gif](https://i.imgur.com/hGF6UzE.gif)

6. Paste your Discord token and your API Keys in the selected fields of the .env file. Remember about the quotation marks!

![image/gif](https://i.imgur.com/oeVaxVQ.gif)

7. Run the bot through the launcher via script:
- On Windows, double-click the launcher.
- On MacOS/Linux, run:
`sh ./launcher_linux_macos.sh`

That's it!

Alternatively, in case you experience any issues with the scripts, you can create a new conda environment yourself using provided 'environment.yml' file, by opening Anaconda Prompt and running following commands:
`cd /your/bot/location`
`conda env create -f environment.yml` - only required for the first time
`conda activate discord_bot`
`python main.py`

IMPORTANT: The console needs to be working in the background for the bot to work.

### Multiple Bots

To have multiple bots, copy-paste your Discord Buddy folder and create for it a new application in Discord Developer App. Each bot needs a separate Discord Token and a console running.

### API

My old tutorials on how to get API Keys:
- Claude (paid): https://rentry.org/marinaraclaude
- Gemini (has a free quota): https://rentry.org/marinaraspaghetti

Personally, I recommend grabbing a custom API Key from OpenRouter and using their free models, like DeepSeek:
https://openrouter.ai/models?max_price=0

![image/png](https://i.imgur.com/90rHJyI.png)

## Features

### 🤖 AI Discord Bot

A versatile AI-powered Discord bot that brings genuine interactions, roleplay, and interactive features to your server! Work with other bots, treating them as other people! Created with spicy sauce by marinara_spaghetti. 🍝 And Claude did some coding too, let's be fair (thank you, Claude, very cool).
Use `/help` to see all the commands.

### ✨ Key Features

#### 🧠 Multiple AI Providers

All easily connectable with a single slash command of `/model_set`. Just remember to add those API Keys first.
- Claude (Anthropic) - Who doesn't love Sonnet? Even if 3.7 is better than 4.
- Gemini (Google) - 03-25, we still miss you.
- ChatGPT (OpenAI) - For those rare few masochists out there.
- Custom APIs - Tested with OpenRouter and my friend's proxy. Supports connections to locally run models as well!

#### 💬 Smart Conversations

Decide how many messages are stored in the context with `/history_length`.
- Context-Aware Responses - Remembers your conversations naturally!
- Image & Voice Support - Bots see pictures, GIFs, and voice messages (dependant on provider).
- Bots see edited messages but not the deleted ones.
- Emoji Reactions - Bots can use and react with emojis (supports custom server emojis, including the animated ones!).
- Bots see other (Buddy) bots as users!
- Autonomous Chatting (Free Will) - Decide in which channels bots will trigger autonomously with `/autonomous_set`!

#### 🎭 Personality System

Decide who your bot is with `/personality_create`.

- Custom Personalities - Create, edit, or delete unique bot characters for your server!
- Per-Server Settings - Different personalities for different communities.
- Persistent Across Restarts - Your bot's personality is saved and remembered.

#### 🎮 Conversation Styles

Choose how your bot communicates with `/prompt_set`.

- Conversational - Natural style chat with emojis and casual language.
- Asterisk Roleplay - A roleplaying abomination style that some of you apparently like.
- Narrative - Rich, story-driven responses with detailed descriptions.
- NSFW variants - Uncensored versions for adult communities.

### 🔒 Works Everywhere

#### 💬 Server Channels

- Mention or reply to the bot to start chatting or set up autonomous responses for natural conversation flow.
- Decide in which channel the bot should use a selected prompt style.
- Admin controls for all important settings.

![image/png](https://i.imgur.com/xOqsOkq.png)

#### 📱 Direct Messages

- Full DM support with all features.
- Decide which server's settings to use in DMs.
- Optional auto check-up messages sent by the bot when you're inactive with `/dm_toggle`.
- Full conversation history loading with `/dm_history_toggle`.
- Edit and regenerate bot responses with `/dm_edit_last` and `/dm_regenerate`.

![image/png](https://i.imgur.com/4gUEE5E.png)

#### 🎪 Fun Interactive Commands

- `/kiss` - Give the bot a kiss and see how they react! 💋
- `/hug` - Warm hugs with personality-based responses. 🤗
- `/joke` - Ask for jokes that match the bot's character. 😄
- `/bonk` - Playfully bonk the bot's head. 🔨
- `/bite` - Chomp! See how they handle it. 🧛
- `/affection` - Find out how much the bot likes you based on your chat history. 💕

![image/png](https://i.imgur.com/pZCrwI6.png)

### 🧠 Memory & Context System

#### 📚 Lore System

Add new lore entries with `/lore_add`.

- Server Mode - Add character information about server members.
- DM Mode - Add personal information about yourself for better roleplay.
- Context-Aware - Automatically switches between server and personal lore.

#### 🧠 Memory Bank

Create memories in DMs and servers, either automatically or manually.

- Auto-generated Memories - Bot creates summaries of important conversations with `/memory_generate`.
- Manual Memory Saving - Save specific moments or information manually with `/memory_save`.
- Smart Recall - Bot remembers relevant details when topics come up.
- Separate Storage - Server and DM memories are kept separately.

### ⚙️ Advanced Configuration

#### 🤖 AI Model Management

- Switch between different AI providers per server on the fly.
- Adjust creativity levels (temperature settings) with `/temperature_set`.
- Real-time provider status checking with `/model_info`.

#### 🛠️ Utility Features

- Clear conversation history with `/clear`.
- Delete bot messages in bulk with `/delete_messages`
- Set custom bot activities and status with `/status_set` and `/activity`.
- Change bot nickname and avatar (admin only) with `/bot_avatar_set` and `/bot_name_set`.

### 📊 Admin Controls

#### 👑 Server Management

- Administrator-only configuration commands.
- DM enable/disable for server members with `/dm_enable`.

#### 🔒 Privacy Features

- No conversation logging to files.
- Optional DM functionalities.
- User-controlled data settings.

### 🚀 Getting Started

- Invite the bot to your server.
- Mention the bot (@botname) to start chatting.
- Use `/help` to see all available commands.
- Set up your preferences with `/model_set`, `/personality_create`, and `/prompt_set`.
- Have fun!

### 💝 Support the Developer

Enjoying the bot? Consider supporting the creator! Every donation helps!
- ☕ Ko-fi: https://ko-fi.com/spicy_marinara

## Details

### Commercial Use

If you want to discuss commercial use or something, hit me up first. That would be very kind of you, thank you!

### Contact

Discord: marinara_spaghetti

### Credits

Special thanks to:
- Claude Sonnet for fixing bugs and setting the groundwork.
- Kuc0 for helping me with the setup scripts.
- Heni for licensing help.
- Crystal for her proxy.
- Dottore for motivating me to finish this.
- Rhy (and his crew), Xixicar, Akiki, Crow, Kiki, Shadow The Shagnus, Vynocchi, and Bun for testing the bots and supporting me.
- My parents for not disinheriting me.
- And you!

![image/png](https://i.imgur.com/nTZ4E4G.png)
