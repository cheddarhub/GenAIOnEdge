// This is a simple client for dllama-api.
//
// Usage:
//
// 1. Start the server, how to do it is described in the `src/apps/dllama-api/README.md` file.
// 2. Run this script: `node examples/chat-api-client.js`

const HOST = process.env.HOST ? process.env.HOST : '127.0.0.1';
const PORT = process.env.PORT ? Number(process.env.PORT) : 9990;

async function chat(messages, maxTokens) {
    try {
        const response = await fetch(`http://${HOST}:${PORT}/v1/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            messages,
            temperature: 0.7,
            stop: ['<|eot_id|>'],
            max_tokens: maxTokens
        }),
        timeout: 1000000
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
} catch (error) {
    console.error('Fetch error:', error);
    throw error; // Rethrow the error for further handling
}
}

async function ask(system, user, maxTokens) {
    console.log(`> system: ${system}`);
    console.log(`> user: ${user}`);
    const response = await chat([
        {
            role: 'system',
            content: system
        },
        {
            role: 'user',
            content: user
        }
    ], maxTokens);
    console.log(response.usage);
    console.log(response.choices[0].message.content);
}

async function main() {
    await ask('You are an excellent math teacher.', 'What is 1 + 2?', 128);
    await ask('You are a romantic.', 'Where is Europe?', 128);
}

main();