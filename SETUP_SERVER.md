# Setting Up Local Pokémon Showdown Server

`poke-env` requires a Pokémon Showdown server to run battles. This guide shows you how to set up a local server.

## Prerequisites

1. **Node.js** (version 14 or higher)
   - Download from: https://nodejs.org/
   - Verify installation: `node --version`

2. **Git** (to clone the repository)
   - Download from: https://git-scm.com/

## Step 1: Clone Pokémon Showdown Repository

```bash
# Navigate to your project directory
cd "C:\Users\Gusma\OneDrive\Desktop\cs 3346"

# Clone the repository
git clone https://github.com/smogon/pokemon-showdown.git

# Navigate into the directory
cd pokemon-showdown
```

## Step 2: Install Dependencies

```bash
# Install Node.js dependencies
npm install
```

## Step 3: Configure Server

```bash
# Copy example config
copy config\config-example.js config\config.js
```

**Note**: On Windows PowerShell, you might need to use:
```powershell
Copy-Item config\config-example.js config\config.js
```

## Step 4: Start the Server

```bash
# Start the server (runs on ws://localhost:8000 by default)
node pokemon-showdown start --no-security
```

**Important**: The `--no-security` flag disables some security features. Only use this for local development.

You should see output like:
```
Pokémon Showdown server running on ws://localhost:8000
```

**Keep this terminal window open** - the server needs to keep running while you train your agent.

## Step 5: Run Your Training Script

In a **new terminal window** (keep the server running):

```bash
# Activate your virtual environment
cd "C:\Users\Gusma\OneDrive\Desktop\cs 3346\cs3346FinalProject"
.\venv\Scripts\Activate.ps1

# Run training
python -m src.trainer
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, you can change it:

1. Edit `config/config.js` in the pokemon-showdown directory
2. Find `port: 8000` and change it to another port (e.g., `8001`)
3. Update `LocalhostServerConfiguration` in your code if needed

### Server Won't Start

- Make sure Node.js is installed: `node --version`
- Make sure you ran `npm install` in the pokemon-showdown directory
- Check that port 8000 is not blocked by firewall

### Connection Refused Error

- Make sure the server is running (check the terminal where you started it)
- Verify the server is listening on `ws://localhost:8000`
- Make sure you're using `LocalhostServerConfiguration` in your code

## Alternative: Use Public Server (Not Recommended)

If you don't want to set up a local server, you can use the public Pokémon Showdown server, but:
- It's slower
- Has rate limits
- Not recommended for training

To use public server, remove `server_configuration=LocalhostServerConfiguration` from player initialization.

## Quick Start Script (Optional)

Create a file `start_server.bat` in your project root:

```batch
@echo off
cd ..\pokemon-showdown
node pokemon-showdown start --no-security
```

Then double-click to start the server quickly.

