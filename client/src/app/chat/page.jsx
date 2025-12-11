'use client'

import React, { useState, useEffect } from 'react';
import { LayoutGrid, List, MessageSquare, Bot, Zap, Search, Plus, Send, Sparkles } from 'lucide-react';

export default function ChatPage() {
  const [activeSession, setActiveSession] = useState('hii');
  const [message, setMessage] = useState('Explain quantum computing in simple terms');
  const [isProcessing, setIsProcessing] = useState(false);
  const [userQuery, setUserQuery] = useState('');
  
  const sessions = [
    { id: 1, title: 'hii', agents: 0, time: '01:02 AM' },
    { id: 2, title: 'hello', agents: 2, time: '12:45 AM' },
    { id: 3, title: 'test', agents: 1, time: 'Yesterday' },
  ];

  const agents = [
    { 
      id: 1, 
      name: 'Gemini 2.5 Pro', 
      role: 'Planner',
      icon: 'https://static.vecteezy.com/system/resources/previews/055/687/063/non_2x/circle-gemini-google-icon-symbol-logo-free-png.png',
      color: 'from-purple-500 to-indigo-500',
      status: 'Thinking...',
      progress: 80
    },
    { 
      id: 2, 
      name: 'Grok 2', 
      role: 'Researcher',
      icon: 'https://images.seeklogo.com/logo-png/60/2/groq-icon-logo-png_seeklogo-605779.png',
      color: 'from-cyan-400 to-blue-500',
      status: 'Researching...',
      progress: 60
    },
    { 
      id: 3, 
      name: 'DeepSeek R1', 
      role: 'Scoring AI',
      icon: 'https://crystalpng.com/wp-content/uploads/2025/01/deepseek-logo-03.png',
      color: 'from-emerald-400 to-green-500',
      status: 'Evaluating...',
      progress: 40
    },
  ];

  const handleSendMessage = (e) => {
    e?.preventDefault();
    if (!message.trim() || isProcessing) return;
    
    setIsProcessing(true);
    setUserQuery(message);
    
    // Reset processing state after a short delay to show loading state
    setTimeout(() => {
      setIsProcessing(false);
    }, 3000);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSendMessage(e);
    }
  };

  // AI Response Panel Component
  const AIResponsePanel = ({ query }) => {
    const [responses, setResponses] = useState({
      gemini: { text: '', isLoading: false, error: null },
      grok: { text: '', isLoading: false, error: null },
      deepsig: { text: '', isLoading: false, error: null }
    });
    const [activeTab, setActiveTab] = useState('gemini');

    useEffect(() => {
      if (!query) return;

      // Make API calls for all models
      Object.keys(responses).forEach(model => {
        fetchAIResponse(model, query);
      });
    }, [query]);

    const fetchAIResponse = async (model, query) => {
      setResponses(prev => ({ ...prev, [model]: { ...prev[model], isLoading: true, error: null } }));
      
      try {
        const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI1MzQ1YTllZS04MDJmLTRkYzQtYjQ5Yi1lOTk3ODQ4MjJkYjIiLCJpYXQiOjE3NjU0MDM2NDksImV4cCI6MTc2NjI2NzY0OX0.CHkMm5lqu7Au4phThALhS3wt2i389sBAOkaeWwMc0t0";
        
        const response = await fetch('https://lunnaa.vercel.app/api/proxy/chat/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
            'Accept': 'text/event-stream',
          },
          body: JSON.stringify({ 
            prompt: query, 
            options: { 
              includeYouTube: true, 
              includeImageSearch: true,
              model: model
            } 
          }),
        });

        if (!response.ok) {
          throw new Error(`Error from ${model} API`);
        }

        // Process the streamed response
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let result = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data:')) {
              try {
                const data = JSON.parse(line.slice(5).trim());
                if (data.text) {
                  result += data.text;
                  setResponses(prev => ({ 
                    ...prev, 
                    [model]: { 
                      ...prev[model], 
                      text: result,
                      isLoading: false 
                    } 
                  }));
                }
              } catch (e) {
                console.error('Error parsing stream data:', e);
              }
            }
          }
        }
      } catch (error) {
        console.error(`[${model}] Error:`, error);
        setResponses(prev => ({ 
          ...prev, 
          [model]: { 
            ...prev[model], 
            error: error.message, 
            isLoading: false 
          } 
        }));
      }
    };

    const renderResponse = (model) => {
      const { text, isLoading, error } = responses[model];
      
      return (
        <div className="h-64 overflow-y-auto p-4">
          {isLoading ? (
            <div className="flex space-x-1 justify-center items-center h-full">
              {[1, 2, 3].map(i => (
                <div 
                  key={i}
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            <div className="prose prose-sm max-w-none">
              <div dangerouslySetInnerHTML={{ __html: text }} />
            </div>
          )}
        </div>
      );
    };

    return (
      <div className="border rounded-lg overflow-hidden mt-4">
        <div className="flex border-b">
          {Object.keys(responses).map((model) => (
            <button
              key={model}
              className={`px-4 py-2 font-medium ${
                activeTab === model
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab(model)}
            >
              {model.charAt(0).toUpperCase() + model.slice(1)}
            </button>
          ))}
        </div>
        {renderResponse(activeTab)}
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-background text-foreground font-body">
      {/* Sidebar */}
      <div className="w-80 bg-card border-r border-border/50 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-border/50">
          <h1 className="text-2xl font-bold font-display bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            AI Collective Arena
          </h1>
          <p className="text-sm text-muted-foreground mt-2">
            Where AI Agents Think, Judge, and Evolve Together
          </p>
        </div>

        {/* Session History */}
        <div className="p-4 border-b border-border/50">
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-sm font-semibold text-muted-foreground">Session History</h2>
            <div className="flex space-x-2">
              <button className="p-1.5 rounded-md hover:bg-accent/10 text-muted-foreground hover:text-accent transition-colors">
                <LayoutGrid className="w-4 h-4" />
              </button>
              <button className="p-1.5 rounded-md hover:bg-accent/10 text-muted-foreground hover:text-accent transition-colors">
                <List className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="space-y-2">
            {sessions.map(session => (
              <div 
                key={session.id}
                className={`p-3 rounded-lg cursor-pointer flex justify-between items-center transition-all duration-200 ${
                  activeSession === session.title 
                    ? 'bg-gradient-to-r from-primary/10 to-accent/5 border border-primary/20 shadow-glow' 
                    : 'hover:bg-accent/5 border border-transparent hover:border-accent/10'
                }`}
                onClick={() => setActiveSession(session.title)}
              >
                <div>
                  <p className="font-medium text-foreground">{session.title}</p>
                  <p className="text-xs text-muted-foreground">{session.agents} agents collaborated</p>
                </div>
                <span className="text-xs text-muted-foreground">{session.time}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Arena Agents */}
        <div className="p-4 flex-1 overflow-y-auto">
          <h2 className="text-sm font-semibold text-muted-foreground mb-3">Arena Agents</h2>
          <div className="space-y-2">
            {agents.map(agent => (
              <div 
                key={agent.id} 
                className="group relative p-3 rounded-lg border border-border/30 hover:border-primary/30 transition-all duration-200 bg-card/50 backdrop-blur-sm"
              >
                <div className="flex items-center space-x-3">
                  <div className="p-1.5">
                    <img 
                      src={agent.icon}
                      alt={agent.name}
                      className="w-8 h-8 object-contain"
                    />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-foreground truncate">{agent.name}</p>
                    <p className="text-xs text-muted-foreground">{agent.role}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Buttons */}
        <div className="p-4 border-t border-border/50 space-y-3 bg-card/30 backdrop-blur-sm">
          <button className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground py-2.5 px-4 rounded-lg font-medium transition-all duration-200 shadow-glow hover:shadow-glow-lg">
            <Zap className="w-4 h-4" />
            <span>Analyzer</span>
          </button>
          <button className="w-full flex items-center justify-center space-x-2 bg-accent/5 hover:bg-accent/10 border border-border/30 text-foreground py-2.5 px-4 rounded-lg font-medium transition-all duration-200">
            <MessageSquare className="w-4 h-4" />
            <span>Critic</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col bg-background">
        {/* Top Bar */}
        <div className="h-16 border-b border-border/30 flex items-center px-8 space-x-8 overflow-x-auto bg-card/30 backdrop-blur-sm">
          {['Synthesizing information...', 'Validating logical consistency...', 'Formulating clear explanations...'].map((task, index) => (
            <div key={index} className="flex items-center space-x-3 whitespace-nowrap group">
              <div className="relative w-3 h-3">
                <div className="absolute inset-0 bg-primary rounded-full animate-ping opacity-75" />
                <div className="absolute inset-0.5 bg-primary rounded-full" />
              </div>
              <span className="text-sm font-medium text-foreground/90 group-hover:text-primary transition-colors">
                {task}
              </span>
              <span className="text-xs text-muted-foreground font-mono">0:0{index + 1}</span>
            </div>
          ))}
          <button className="ml-auto flex items-center space-x-2 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground py-2 px-4 rounded-lg text-sm font-medium transition-all duration-200 shadow-glow hover:shadow-glow-lg">
            <Plus className="w-4 h-4" />
            <span>New Arena Session</span>
          </button>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-hidden relative">
          {/* Session Info */}
          <div className="p-6 pb-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold font-display text-foreground">{activeSession}</h2>
                <p className="text-sm text-muted-foreground">
                  Arena Session • {sessions.find(s => s.title === activeSession)?.time || '01:02 AM'}
                </p>
              </div>
            </div>
          </div>

          {/* AI Response Panel */}
          <div className="px-6 pb-4">
            {userQuery && <AIResponsePanel query={userQuery} />}
          </div>
        </div>

        {/* Input Area */}
        <div className="p-6 border-t border-border/30 bg-card/30 backdrop-blur-sm">
          <form onSubmit={handleSendMessage} className="relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-muted-foreground">
              <MessageSquare className="w-4 h-4" />
            </div>
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full bg-background border border-border/50 hover:border-primary/50 focus:border-primary/70 rounded-xl py-3 pl-10 pr-32 focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all duration-200 text-foreground placeholder:text-muted-foreground/50"
              placeholder="Ask anything..."
              disabled={isProcessing}
            />
            <button 
              type="submit"
              className={`absolute right-2 top-1/2 -translate-y-1/2 flex items-center space-x-1.5 px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                isProcessing 
                  ? 'bg-primary/10 text-primary' 
                  : 'bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground shadow-glow hover:shadow-glow-lg'
              }`}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <>
                  <span>Processing</span>
                  <div className="flex space-x-1">
                    {[1, 2, 3].map((i) => (
                      <div 
                        key={i}
                        className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce"
                        style={{ animationDelay: `${i * 0.15}s` }}
                      />
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <span>Send</span>
                  <Send className="w-4 h-4" />
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
