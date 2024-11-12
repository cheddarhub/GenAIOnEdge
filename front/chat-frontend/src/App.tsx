//import React, { useState } from 'react';
import { useState } from 'react';  // Only import what you need (useState)
import './App.css';

// Define response type
interface ChatResponse {
  response: string;
  error?: string;
}

function App() {
  const [model, setModel] = useState<string>("gpt_neo");
  const [message, setMessage] = useState<string>("");
  const [response, setResponse] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);



  // Handle sending the message to the FastAPI backend
  const handleSendMessage = async () => {
    if (!message) {
      alert("Please enter a message.");
      return;
    }
    //http://proxy:8000/
    setLoading(true);
    try {
      const res = await fetch("http://10.41.119.56:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          model: model,
        }),
      });

      const data: ChatResponse = await res.json();
      if (data.error) {
        setResponse(`Error: ${data.error}`);
      } else {
        console.log(data.response);
        setResponse(data.response);
      }
    } catch (error) {
      setResponse("Failed to communicate with the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="sidebar">
        <select className="model-select" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="gpt_neo">GPT-Neo</option>
          <option value="mistral">Mistral</option>
          <option value="internlm">internLM</option>
          <option value="llama">Llama</option>
        </select>
        {/* Add more sidebar content here */}
      </div>
      <div className="chat-container">
        <div className="messages">
          {/* Display chat messages here */}
          {response && <div className="response">{response}</div>}
        </div>
        <div className="input-area">
          <textarea
            className="message-input"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message here..."
          />
          <button className="send-button" onClick={handleSendMessage} disabled={loading}>
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
