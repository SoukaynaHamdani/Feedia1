import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Mic, Square, ArrowLeft, Loader2, Volume2, Send } from 'lucide-react';
import useAppStore from '../store/useAppStore';
import { speakText, stopSpeaking } from '../services/aiService';

export default function VoiceChat() {
  const { setView, language } = useAppStore();

  const ttsAudioRef = useRef(null);
  // Default to the language picked in the app: auto-detection mistakes Darija for English.
  const [voiceLang, setVoiceLang] = useState(
    language === 'ar' || language === 'darija' ? 'darija'
      : language === 'tz' || language === 'tamazight' ? 'tamazight'
      : language === 'fr' ? 'fr' : 'en'
  );

  const playMessage = async (msg) => {
    stopSpeaking();
    if (ttsAudioRef.current) { ttsAudioRef.current.pause(); ttsAudioRef.current = null; }
    const ttsLang = msg.lang || (voiceLang !== 'auto' ? voiceLang : language === 'ar' ? 'darija' : language === 'tz' ? 'tamazight' : language);

    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: msg.text, language: ttsLang }),
      });
      if (!res.ok) throw new Error(`TTS HTTP ${res.status}`);
      const audio = new Audio(URL.createObjectURL(await res.blob()));
      ttsAudioRef.current = audio;
      await audio.play();
      return;
    } catch (err) {
      console.warn('Server TTS unavailable, using browser voice:', err);
    }
    speakText(msg.text, ttsLang).catch(err => console.warn('TTS failed:', err));
  };

  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [inputText, setInputText] = useState('');
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingRequestedRef = useRef(false);

  useEffect(() => {
    // Initial greeting
    setMessages([
      {
        id: 1,
        sender: 'ai',
        text: language === 'ar' || language === 'darija' 
          ? "أهلاً بك! شنو نوع البهائم اللي عندك و شحال العدد ديالهم؟ (يمكن ليك تكتب أو تهضر معايا بالصوت)" 
          : language === 'tz' || language === 'tamazight'
          ? "ⴰⵣⵓⵍ! ⵎⴰⵏ ⵉⵎⵓⴷⴰⵔ ⵖⵓⵔⴽ ⴷ ⵎⵏⵛⴽ ⴰⵢⵍⵍⴰⵏ? (ⵜⵣⵎⵔⴷ ⴰⴷ ⵜⴰⵔⴰⴷ ⵏⵖ ⴰⴷ ⵜⵙⴰⵡⵍⴷ)"
          : "Hello! What animal do you want to feed, and how many do you have? (You can type or speak to me)",
      }
    ]);
    return () => stopSpeaking();
  }, [language]);

  const handleTextSend = async (e) => {
    e?.preventDefault();
    if (!inputText.trim()) return;

    const userText = inputText.trim();
    setInputText('');
    
    const newUserMessage = {
      id: Date.now(),
      sender: 'user',
      text: userText,
    };
    
    setMessages(prev => [...prev, newUserMessage]);
    setIsProcessing(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userText, language: voiceLang })
      });

      if (!response.ok) throw new Error('Failed to process text');
      const data = await response.json();
      
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'ai',
        text: data.result,
        lang: data.language,
        canPlay: true
      }]);
    } catch (error) {
      console.error('API Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        sender: 'ai',
        text: "Sorry, I encountered an error processing your request."
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const startRecording = async () => {
    recordingRequestedRef.current = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!recordingRequestedRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      const mimeType = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
      ].find(type => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined
      );
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        });
        stream.getTracks().forEach(track => track.stop());
        mediaRecorderRef.current = null;
        audioChunksRef.current = [];
        await handleAudioSend(audioBlob);
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      recordingRequestedRef.current = false;
      console.error('Error accessing microphone:', error);
      alert('Could not access the microphone. Please check your permissions.');
    }
  };

  const stopRecording = () => {
    if (!recordingRequestedRef.current) return;
    recordingRequestedRef.current = false;
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === 'recording') {
      recorder.stop();
    }
    setIsRecording(false);
  };

  const handleAudioSend = async (audioBlob) => {
    if (!audioBlob.size) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        sender: 'ai',
        text: 'No audio was captured. Please hold the microphone button while speaking, then release it to send.',
      }]);
      return;
    }
    setIsProcessing(true);
    
    // Create an object URL for the user's audio to show in chat
    const userAudioUrl = URL.createObjectURL(audioBlob);
    
    const newUserMessage = {
      id: Date.now(),
      sender: 'user',
      audioUrl: userAudioUrl,
      text: '...', // Will be replaced by transcription
    };
    
    setMessages(prev => [...prev, newUserMessage]);

    // Send to backend
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('language', voiceLang);

    try {
      const response = await fetch('/api/voice-chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `Voice request failed (${response.status}).`
        );
      }

      const data = await response.json();
      
      // Update user message with transcription
      setMessages(prev => prev.map(msg => 
        msg.id === newUserMessage.id 
          ? { ...msg, text: data.transcription } 
          : msg
      ));

      // Add AI response
      const newAiMessage = {
        id: Date.now() + 1,
        sender: 'ai',
        text: data.response,
        lang: data.language,
        audioBase64: data.audio_base64
      };
      
      setMessages(prev => [...prev, newAiMessage]);
      
      // Autoplay the reply (Darija TTS server, falling back to browser/cloud voice)
      playMessage(newAiMessage);

    } catch (error) {
      console.error('API Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        sender: 'ai',
        text: error instanceof Error
          ? `Voice request failed: ${error.message}`
          : "Sorry, I encountered an error processing your voice."
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col h-[80vh] w-full max-w-2xl mx-auto bg-white rounded-[2rem] shadow-2xl overflow-hidden mt-8 border border-slate-100">
      {/* Header */}
      <div className="bg-agricultural-green text-white p-4 flex items-center justify-between shadow-md z-10 relative">
        <button 
          onClick={() => setView('home')}
          className="p-2 hover:bg-white/20 rounded-full transition-colors"
        >
          <ArrowLeft className="w-6 h-6" />
        </button>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm">
            <Volume2 className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-lg">AI Voice Assistant</h2>
            <p className="text-white/80 text-xs">Darija · Français · English</p>
          </div>
        </div>
        <div className="w-10"></div> {/* Spacer for centering */}
      </div>

      {/* Language bar */}
      <div className="flex items-center justify-center gap-2 px-4 py-2 bg-white border-b border-slate-100 text-sm">
        {[
          ['auto', 'Auto'],
          ['darija', 'الدارجة'],
          ['fr', 'Français'],
          ['en', 'English'],
        ].map(([code, label]) => (
          <button
            key={code}
            onClick={() => setVoiceLang(code)}
            className={`px-3 py-1 rounded-full border transition-colors ${
              voiceLang === code
                ? 'bg-agricultural-green text-white border-agricultural-green'
                : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50">
        {messages.map((msg) => (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            key={msg.id} 
            className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div 
              className={`max-w-[80%] rounded-2xl p-4 shadow-sm ${
                msg.sender === 'user' 
                  ? 'bg-agricultural-green text-white rounded-br-none' 
                  : 'bg-white text-slate-800 rounded-bl-none border border-slate-100'
              }`}
            >
              {msg.audioUrl && (
                <audio controls src={msg.audioUrl} className="h-8 mb-2 w-48" />
              )}
              <div className="flex items-start gap-3">
                <p className={`flex-1 whitespace-pre-line text-sm md:text-base ${msg.sender === 'user' ? 'text-white/90' : 'text-slate-700'} leading-relaxed`}>
                  {msg.text}
                </p>
                {msg.sender === 'ai' && msg.text && (
                  <button
                    onClick={() => playMessage(msg)}
                    title="Listen"
                    className="shrink-0 w-9 h-9 flex items-center justify-center rounded-full bg-agricultural-green text-white hover:opacity-90 transition-opacity"
                  >
                    <Volume2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        ))}
        {isProcessing && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-start"
          >
            <div className="bg-white rounded-2xl rounded-bl-none p-4 shadow-sm border border-slate-100 flex items-center gap-3">
              <Loader2 className="w-5 h-5 text-agricultural-green animate-spin" />
              <span className="text-slate-500 text-sm">Processing voice...</span>
            </div>
          </motion.div>
        )}
      </div>

      {/* Footer / Controls */}
      <div className="p-4 bg-white border-t border-slate-100 flex flex-col sm:flex-row items-center gap-4">
        {/* Text Input */}
        <form onSubmit={handleTextSend} className="flex-1 w-full flex items-center bg-slate-50 border border-slate-200 rounded-full px-4 py-2 focus-within:border-agricultural-green focus-within:ring-2 focus-within:ring-agricultural-green/20 transition-all">
          <input 
            type="text" 
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your message here..."
            className="flex-1 bg-transparent border-none focus:outline-none text-slate-700 placeholder:text-slate-400"
            disabled={isProcessing}
          />
          <button 
            type="submit" 
            disabled={!inputText.trim() || isProcessing}
            className="p-2 text-agricultural-green hover:bg-agricultural-green/10 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>

        <div className="relative flex items-center justify-center shrink-0">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onPointerDown={startRecording}
            onPointerUp={stopRecording}
            onPointerLeave={stopRecording}
            disabled={isProcessing}
            className={`relative flex items-center justify-center w-14 h-14 sm:w-16 sm:h-16 rounded-full shadow-lg transition-colors ${
              isRecording 
                ? 'bg-red-500 hover:bg-red-600' 
                : isProcessing 
                  ? 'bg-slate-300 cursor-not-allowed'
                  : 'bg-agricultural-green hover:bg-agricultural-green-dark'
            }`}
          >
            {isRecording && (
              <span className="absolute inset-0 rounded-full border-4 border-red-500/30 animate-ping" />
            )}
            {isRecording ? (
              <Square className="w-6 h-6 sm:w-8 sm:h-8 text-white fill-white" />
            ) : (
              <Mic className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
            )}
          </motion.button>
          {isRecording && (
            <span className="absolute -top-6 text-xs text-red-500 font-bold animate-pulse whitespace-nowrap">
              Recording...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
