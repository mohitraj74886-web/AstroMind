import { useRef, useState, useEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger);

// 1. Base Domain Configuration
const HF_SPACE_URL = "https://moss2110-astromind.hf.space";

// 2. Clear Feature Subrouters
const LLM_API_URL = `${HF_SPACE_URL}/llm`;     // Maps /llm/start_session, /llm/process_response
const VOICE_API_URL = `http://127.0.0.1:8000`;   // Maps /voice/diagnose
const ECHOES_API_URL = "http://127.0.0.1:8003";  // Local physics-delay node

const AstroMindHome = () => {
  const videoRef = useRef(null);

  // Layout Transition State
  const [showChat, setShowChat] = useState(false);
  const [showEchoesInbox, setShowEchoesInbox] = useState(false);

  // Live Session Registry States
  const [sessionId, setSessionId] = useState(null);
  const [currentPhase, setCurrentPhase] = useState("intro");
  const [riskLevel, setRiskLevel] = useState("low");
  const [biometricAnomaly, setBiometricAnomaly] = useState(false);
  const [sessionSummary, setSessionSummary] = useState(null);

  // Feature Section Refs
  const containerRef = useRef(null);
  const stickyRef = useRef(null);
  const fillBarRef = useRef(null);
  const glowDotRef = useRef(null);
  const [activeFeature, setActiveFeature] = useState(0);
  const [isYoYoMode, setIsYoYoMode] = useState(true);

  // How It Works Section Refs
  const howItWorksContainer = useRef(null);
  const howItWorksSticky = useRef(null);
  const centerMascotRef = useRef(null);
  const cardPurpleRef = useRef(null);
  const cardPinkRef = useRef(null);
  const cardOrangeRef = useRef(null);
  const headingRef = useRef(null);

  // 💬 AI Chat Interface State
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isAiTyping, setIsAiTyping] = useState(false);
  const [isVoiceMuted, setIsVoiceMuted] = useState(false);
  const chatEndRef = useRef(null);
  const [isListening, setIsListening] = useState(false);

  // 🛰️ Echoes Communication State
  const [echoesInbox, setEchoesInbox] = useState([]);
  const [transitCount, setTransitCount] = useState(0);

  // 🎙️ Media Hardware Streams for Hugging Face Diagnostics
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Initialize live session on port 8004
  const startAstroMindSession = async () => {
    try {
      setIsAiTyping(true);
      const response = await fetch(`${LLM_API_URL}/start_session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          astronaut_id: "cmdr_tanaka_01",
          mission_day: 42
        })
      });

      if (!response.ok) throw new Error("Session handshake rejected.");
      const data = await response.json();

      setSessionId(data.session_id);
      setCurrentPhase(data.session_phase);

      setMessages([{ id: Date.now(), sender: 'ai', text: data.first_question }]);
      setIsAiTyping(false);
      speakText(data.first_question);
    } catch (err) {
      console.error("Failed to connect with llm_api:", err);
      setMessages([{ id: Date.now(), sender: 'ai', text: "Telemetry link dropped. Ensure your local LLM engine is active over port 8004." }]);
      setIsAiTyping(false);
    }
  };

  // Trigger initialization sequence when entering terminal layout
  useEffect(() => {
    if (showChat && !sessionId) {
      startAstroMindSession();
    }
  }, [showChat]);

  // Fetch verified arrivals from the echoes speed-of-light queue on port 8003
  const checkEchoesSubspaceChannel = async () => {
    try {
      const response = await fetch(`${ECHOES_API_URL}/echoes/inbox/Commander`);
      if (!response.ok) throw new Error("Channel network exception.");
      const data = await response.json();

      setEchoesInbox(data.inbox);
      setTransitCount(data.messages_in_transit);
    } catch (err) {
      console.error("Echoes endpoint connection failure:", err);
    }
  };

  useEffect(() => {
    if (showEchoesInbox) {
      checkEchoesSubspaceChannel();
    }
  }, [showEchoesInbox]);

  // Handle live form pipeline transitions to llm_api processing
  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!inputMessage.trim() || !sessionId) return;

    const userRawText = inputMessage;
    setMessages(prev => [...prev, { id: Date.now(), sender: 'user', text: userRawText }]);
    setInputMessage('');
    setIsAiTyping(true);

    try {
      const response = await fetch(`${LLM_API_URL}/process_response`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          transcribed_text: userRawText,
          astronaut_id: "cmdr_tanaka_01"
        })
      });

      if (!response.ok) throw new Error("Processing response failed.");
      const data = await response.json();

      setCurrentPhase(data.session_phase);
      setRiskLevel(data.risk_level);
      setBiometricAnomaly(data.biometric_anomaly_detected);

      const aiResponseMsg = { id: Date.now() + 1, sender: 'ai', text: data.llm_response };
      setMessages(prev => [...prev, aiResponseMsg]);
      speakText(data.llm_response);

      if (data.next_question) {
        setTimeout(() => {
          setMessages(prev => [...prev, { id: Date.now() + 2, sender: 'ai', text: data.next_question }]);
          speakText(data.next_question);
        }, 1500);
      }

      if (data.is_complete) {
        setSessionSummary(data.session_summary);
      }

    } catch (err) {
      console.error("Transmission relay error:", err);
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'ai', text: "Signal degradation error. Ingestion frame unfulfilled." }]);
    } finally {
      setIsAiTyping(false);
    }
  };

  // Speech to Text (Voice Input) Engine + HF Space Audio Capture Muxer
  // Pure Web Audio API WAV Capture & Transmission Engine
  const startVoiceInput = async () => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = audioContext.createMediaStreamSource(audioStream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      const leftChannelBuffer = [];
      setIsListening(true);
      window.speechSynthesis.cancel();

      processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        leftChannelBuffer.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      // Provide a way to intercept the Web Speech API transcription loop cleanly
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        setInputMessage(event.results[0][0].transcript);
      };

      recognition.onerror = () => setIsListening(false);

      recognition.onend = async () => {
        setIsListening(false);

        // 1. Disconnect and freeze the live audio capture pipeline
        processor.disconnect();
        source.disconnect();
        audioStream.getTracks().forEach(track => track.stop());
        await audioContext.close();

        // 2. Flatten our captured float arrays into a unified sequence
        const flattenedBuffer = flattenChannelBuffers(leftChannelBuffer);

        // 3. Compile a true standard 16-bit PCM WAV Binary ArrayBuffer
        const wavBuffer = createWavDataView(flattenedBuffer, audioContext.sampleRate);
        const audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });

        // 4. Ship the raw valid WAV payload down the pipe
        const voiceFormData = new FormData();
        voiceFormData.append("file", audioBlob, "cabin_transmission.wav");

        console.log("Transmitting verified native WAV audio packet to backend...");
        try {
          const hfResponse = await fetch(`${VOICE_API_URL}/diagnose`, {
            method: "POST",
            body: voiceFormData
          });

          if (!hfResponse.ok) {
            const errorDetails = await hfResponse.json();
            console.error("❌ Validation failure tracking:", errorDetails);
            return;
          }

          const hfDiagnosticData = await hfResponse.json();
          console.log("🌲 [VOX ANALYSIS SUCCESS]:", hfDiagnosticData);
        } catch (hfErr) {
          console.error("Diagnostic uplink processing failure:", hfErr);
        }
      };

      recognition.start();

    } catch (deviceErr) {
      console.error("Failed to securely initiate standard capture audio parameters:", deviceErr);
      alert("Could not access physical capture vitals.");
    }
  };

  // --- WAV AUDIO COMPILATION HELPER FUNCTIONS ---

  const flattenChannelBuffers = (bufferList) => {
    let totalLength = 0;
    for (let i = 0; i < bufferList.length; i++) {
      totalLength += bufferList[i].length;
    }
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (let i = 0; i < bufferList.length; i++) {
      result.set(bufferList[i], offset);
      offset += bufferList[i].length;
    }
    return result;
  };

  const createWavDataView = (samples, sampleRate) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw PCM = 1) */
    view.setUint16(20, 1, true);
    /* channel count (Mono = 1) */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample (16-bit PCM) */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    // Write actual PCM audio sequence samples down into the data array view
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return buffer;
  };

  const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  // Text to Speech Engine (TTS)
  const speakText = (text) => {
    if (isVoiceMuted || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const premiumVoice = voices.find(v => v.name.includes('Google US English') || v.name.includes('Natural'));
    if (premiumVoice) utterance.voice = premiumVoice;
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  // Auto Scroll Chat to bottom
  useEffect(() => {
    if (showChat) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, showChat, isAiTyping]);

  // Video Loop Handling
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let animationFrameId;
    let lastTime = performance.now();
    let direction = 1;

    const updateLoop = (now) => {
      const deltaTime = (now - lastTime) / 1000;
      lastTime = now;

      if (isYoYoMode) {
        if (direction === 1) {
          if (video.currentTime >= video.duration - 0.15) {
            direction = -1;
            video.pause();
          }
        } else {
          video.currentTime -= deltaTime;
          if (video.currentTime <= 0.15) {
            direction = 1;
            video.play().catch(() => { });
          }
        }
      }
      animationFrameId = requestAnimationFrame(updateLoop);
    };

    if (isYoYoMode) {
      video.loop = false;
      animationFrameId = requestAnimationFrame(updateLoop);
    } else {
      video.loop = true;
      video.play().catch(() => { });
    }

    return () => cancelAnimationFrame(animationFrameId);
  }, [isYoYoMode]);

  // GSAP Structural Layout Timelines
  useGSAP(() => {
    if (showChat) return;
    ScrollTrigger.create({
      trigger: containerRef.current,
      start: "top top",
      end: "bottom bottom",
      pin: stickyRef.current,
      scrub: true,
    });

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top top",
        end: "bottom bottom",
        scrub: 0.5,
      }
    });

    tl.fromTo(fillBarRef.current, { height: "0%" }, { height: "100%", ease: "none", duration: 3 });
    tl.fromTo(glowDotRef.current, { bottom: "0%" }, { bottom: "100%", ease: "none", duration: 3 }, "<");

    tl.call(() => setActiveFeature(0), null, 0.0);
    tl.call(() => setActiveFeature(1), null, 1.0);
    tl.call(() => setActiveFeature(2), null, 2.0);
  }, { scope: containerRef, dependencies: [showChat] });

  useGSAP(() => {
    if (showChat) return;
    ScrollTrigger.create({
      trigger: howItWorksContainer.current,
      start: "top top",
      end: "bottom bottom",
      pin: howItWorksSticky.current,
      scrub: true,
    });

    const workTl = gsap.timeline({
      scrollTrigger: {
        trigger: howItWorksContainer.current,
        start: "top top",
        end: "bottom bottom",
        scrub: 0.6,
      }
    });

    workTl.to(centerMascotRef.current, { rotation: 90, scale: 1.2, y: -20, ease: "power1.inOut" }, 0)
      .to(headingRef.current, { y: -40, scale: 0.9, opacity: 0.4, ease: "power1.inOut" }, 0)
      .to(cardPurpleRef.current, { x: -100, y: 150, opacity: 0, scale: 0.7, ease: "power1.inOut" }, 0)
      .fromTo(cardPinkRef.current, { x: 0, y: 180, scale: 0.8, opacity: 0.3 }, { x: 180, y: -80, scale: 1, ease: "power1.inOut" }, 0)
      .fromTo(cardOrangeRef.current, { y: 300, x: -150, opacity: 0, scale: 0.5 }, { y: 60, x: -160, opacity: 1, scale: 1, ease: "power1.inOut" }, 0);

    workTl.to(cardPinkRef.current, { x: -80, y: -40, scale: 1.1, zIndex: 50, ease: "power2.inOut", duration: 0.5, opacity: 1 })
      .to(cardOrangeRef.current, { opacity: 0.3, scale: 0.9, ease: "power2.inOut", duration: 0.5 }, "<")
      .to(centerMascotRef.current, { opacity: 1, scale: 0.8, rotate: 180, ease: "power2.inOut", duration: 0.5 }, "<");
  }, { scope: howItWorksContainer, dependencies: [showChat] });


  return (
    <div className="relative min-h-screen bg-black text-lime-100 overflow-x-hidden font-cursive">

      {/* BACKGROUND VIDEO LAYER */}
      <div className="fixed inset-0 w-full h-full z-0 overflow-hidden pointer-events-none">
        <video ref={videoRef} className="absolute top-1/2 left-1/2 w-full h-full object-cover -translate-x-1/2 -translate-y-1/2 opacity-25 mix-blend-overlay" src="./bg.mp4" muted playsInline autoPlay />
      </div>

      {showChat ? (
        /* 💬 THE AI CHAT SYSTEM INTERFACE OVERLAY LAYER */
        <div className="relative z-10 w-full min-h-screen flex flex-col justify-between max-w-4xl mx-auto px-4 py-6 font-mono">
          <header className="flex justify-between items-center border-b border-lime-800/50 pb-4 mb-4 backdrop-blur-md bg-black/20 p-4 rounded-xl">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${riskLevel === 'critical' ? 'bg-red-500 animate-ping' : 'bg-lime-400 animate-pulse'}`} />
              <div>
                <h1 className="text-sm font-bold tracking-wider text-white uppercase">AstroMind Terminal v2.0</h1>
                <p className="text-[10px] text-lime-400/60">PHASE: {currentPhase.toUpperCase()} | STATUS: {riskLevel.toUpperCase()} RISK</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => { setIsVoiceMuted(!isVoiceMuted); if (!isVoiceMuted) window.speechSynthesis.cancel(); }} className={`p-2 rounded-lg border text-xs transition-all ${isVoiceMuted ? 'border-red-900/60 bg-red-950/20 text-red-400' : 'border-lime-800/60 bg-lime-950/20 text-lime-400 hover:bg-lime-900/30'}`}>
                {isVoiceMuted ? '🔇 Muted' : '🔊 Audio ON'}
              </button>
              <button onClick={() => { window.speechSynthesis.cancel(); setSessionId(null); setSessionSummary(null); setShowChat(false); }} className="px-3 py-1 bg-black/40 text-lime-200 text-xs border border-lime-800/80 rounded-lg hover:bg-lime-900/30 transition-all">
                ← Disconnect
              </button>
            </div>
          </header>

          {/* Interactive Messages Display Board Box */}
          <div className="flex-1 overflow-y-auto space-y-4 my-2 pr-2 scrollbar-thin custom-scrollbar max-h-[60vh]">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex flex-col max-w-[80%] ${msg.sender === 'user' ? 'ml-auto items-end' : 'mr-auto items-start'}`}>
                <span className="text-[10px] tracking-widest text-lime-500/40 uppercase mb-1">
                  {msg.sender === 'user' ? '🛰️ You (Transmission)' : '🤖 AstroMind'}
                </span>
                <div className={`p-4 rounded-2xl text-sm leading-relaxed border transition-all duration-300 ${msg.sender === 'user' ? 'bg-lime-950/40 border-lime-500/40 text-lime-200 rounded-tr-none' : 'bg-zinc-900/80 border-zinc-800 text-white rounded-tl-none shadow-[0_0_15px_rgba(163,230,53,0.03)]'}`}>
                  {msg.text}
                </div>
              </div>
            ))}

            {isAiTyping && (
              <div className="flex flex-col max-w-[80%] mr-auto items-start">
                <span className="text-[10px] tracking-widest text-lime-500/40 uppercase mb-1">🤖 AstroMind</span>
                <div className="p-4 rounded-2xl rounded-tl-none bg-zinc-900/40 border border-zinc-800/60 text-lime-400/70 text-sm flex items-center gap-2">
                  <span className="animate-pulse">Analyzing telemetry metrics</span>
                </div>
              </div>
            )}

            {/* Dynamic Clinical Summary Dashboard Component */}
            {sessionSummary && (
              <div className="mt-6 border-2 border-dashed border-lime-500/40 bg-zinc-950/90 rounded-2xl p-6 text-white transition-all duration-500">
                <h3 className="text-xs font-bold font-mono uppercase tracking-widest text-lime-400 mb-2">📋 Compiled Clinical Assessment</h3>
                <p className="text-sm font-sans font-light text-zinc-300 leading-relaxed">{sessionSummary}</p>
                <div className="mt-4 pt-4 border-t border-zinc-800/80 flex justify-between text-[10px] text-zinc-500">
                  <span>LOG ENCRYPTED AND STORED SECURELY</span>
                  <span className={biometricAnomaly ? "text-amber-400 font-bold" : "text-lime-400"}>
                    {biometricAnomaly ? "⚠️ BIOMETRIC ANOMALY FLAGGED" : "✓ BASELINE PHYSIOLOGY STABLE"}
                  </span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input Panel */}
          <form onSubmit={handleSendMessage} className="mt-4 flex gap-2 border-t border-lime-800/30 pt-4 backdrop-blur-md">
            <div className="relative flex-1">
              <input type="text" value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} placeholder={isListening ? "Listening to cabin audio..." : "Type your transmission log here safely..."} className="w-full pl-4 pr-12 py-3 bg-zinc-900/90 border border-lime-800/50 rounded-xl text-white focus:outline-none focus:border-lime-400 transition-all text-sm" disabled={isAiTyping || !!sessionSummary} />
              <button type="button" onClick={startVoiceInput} disabled={isAiTyping || !!sessionSummary} className={`absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg transition-all ${isListening ? 'text-red-400 animate-pulse bg-red-950/30' : 'text-lime-400 hover:bg-lime-900/20'}`}>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3-3.75h6M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" /></svg>
              </button>
            </div>
            <button type="submit" disabled={isAiTyping || !inputMessage.trim() || !!sessionSummary} className="px-6 py-3 bg-lime-400 disabled:bg-lime-900/40 text-[#1b3002] disabled:text-lime-700 font-bold rounded-xl text-sm transition-all shadow-md">
              Transmit →
            </button>
          </form>
        </div>
      ) : showEchoesInbox ? (
        /* 🛰️ THE ECHOES INBOX LAYER (PORT 8003 INTERACTIVE SCREEN) */
        <div className="relative z-10 w-full min-h-screen flex flex-col justify-center max-w-2xl mx-auto px-4 py-12 font-mono">
          <header className="mb-6 flex justify-between items-center border-b border-zinc-800 pb-4">
            <div>
              <h2 className="text-xl font-bold text-white tracking-wide uppercase">Wave Receiver Terminal</h2>
              <p className="text-xs text-zinc-500">IN TRANSIT: {transitCount} DATA PACKETS BOUND FROM EARTH</p>
            </div>
            <button onClick={() => setShowEchoesInbox(false)} className="px-3 py-1 text-xs border border-zinc-700 text-zinc-300 rounded-lg hover:bg-zinc-900 transition-all">
              ← Return To Deck
            </button>
          </header>

          <div className="space-y-4 min-h-[45vh] bg-zinc-950/80 border border-zinc-900 rounded-2xl p-6 shadow-2xl backdrop-blur-md overflow-y-auto">
            {echoesInbox.length === 0 ? (
              <p className="text-center text-xs text-zinc-600 pt-20 uppercase tracking-widest">No wave packets have completed physical velocity time-delay metrics.</p>
            ) : (
              echoesInbox.map((msg) => (
                <div key={msg.id} className="p-4 bg-zinc-900/50 border border-zinc-800/80 rounded-xl space-y-3">
                  <div className="flex justify-between text-[10px] tracking-wider text-zinc-400">
                    <span className="text-lime-400 font-bold uppercase">FROM: {msg.sender}</span>
                    <span>ARRIVED: {new Date(msg.arrival_time_utc).toLocaleTimeString()}</span>
                  </div>
                  <audio controls src={msg.file_url} className="w-full filter invert brightness-95 opacity-75" />
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        /* MAIN LANDING DECK INTERFACE LAYERS */
        <div className="relative z-10 w-full">
          <header className="relative min-h-screen flex flex-col justify-center items-center px-6 text-center max-w-5xl mx-auto font-handwriting">
            <div className="space-y-6">
              <span className="inline-block text-xs tracking-[0.3em] uppercase text-lime-400 font-bold opacity-80">System Initialization</span>
              <h2 className="text-3xl md:text-7xl font-display uppercase tracking-tight text-white leading-none">Your quiet space <br /><span className="text-lime-300">among the stars.</span></h2>
              <p className="text-base md:text-xl font-light text-lime-100/80 max-w-2xl mx-auto leading-relaxed font-charon">Deep space is vast, but you never have to navigate it alone. AstroMind is your private, onboard companion—here to listen, understand, and support your well-being.</p>
              <div className="pt-6 flex flex-col sm:flex-row gap-4 justify-center items-center">
                <button onClick={() => setShowChat(true)} className="w-full sm:w-auto px-8 py-4 bg-lime-400 text-[#1b3002] rounded-full font-bold shadow-lg hover:bg-lime-300 transition-all duration-300">Begin Check-In</button>
                <button onClick={() => setShowEchoesInbox(true)} className="w-full sm:w-auto px-8 py-4 bg-black/40 text-lime-200 border border-lime-800/80 backdrop-blur-md rounded-full font-medium hover:bg-black/60 transition-all duration-300">Listen to Echoes</button>
              </div>
            </div>
          </header>

          <section className="py-24 px-6 max-w-4xl mx-auto text-center font-charon">
            <h2 className="text-3xl md:text-5xl font-display uppercase text-white tracking-wide mb-6">A sanctuary for your mind.</h2>
            <p className="text-base md:text-lg font-light text-lime-100/70 leading-relaxed max-w-2xl mx-auto">Living in isolation requires immense resilience. AstroMind was built to be a steady anchor during your mission. By combining empathetic conversation with a gentle understanding of your body’s natural rhythms, we provide a safe, confidential space for you to process your thoughts, reflect on your days, and find a moment of peace.</p>
          </section>

          <div ref={containerRef} className="relative h-[300vh] w-full font-charon">
            <div ref={stickyRef} className="w-full h-screen flex items-center justify-center bg-black/5 overflow-hidden">
              <div className="max-w-6xl w-full mx-auto px-6 grid grid-cols-1 md:grid-cols-12 items-center gap-8 relative h-full py-12">
                <div className="md:col-span-5 space-y-4 flex flex-col justify-center">
                  <h2 className="text-5xl md:text-7xl font-display uppercase text-white leading-none tracking-tight">What is <br /><span className="text-lime-300">AstroMind?</span></h2>
                </div>
                <div className="md:col-span-2 relative h-[350px] flex items-center justify-center">
                  <div className="absolute inset-y-0 w-[2px] bg-white/10 rounded-full">
                    <div ref={fillBarRef} className="w-full bg-lime-400 origin-bottom absolute bottom-0 left-0 shadow-[0_0_10px_#a3e635]" />
                    <div ref={glowDotRef} className="absolute left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-slate-200 ring-4 ring-lime-500 shadow-[0_0_20px_#a3e635] z-10" />
                  </div>
                </div>
                <div className="md:col-span-5 relative h-[250px] flex flex-col justify-center">
                  <div className={`transition-all duration-500 absolute inset-x-0 space-y-4 ${activeFeature === 0 ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-4 scale-95 pointer-events-none'}`}>
                    <div className="text-6xl font-display text-white/5 select-none leading-none">01</div>
                    <h3 className="text-xl md:text-2xl font-bold text-white tracking-wide">Compassionate Conversation</h3>
                    <p className="text-sm font-light text-lime-200/60 leading-relaxed">Whenever you need a sounding board, AstroMind is ready. No judgment, no rush.</p>
                  </div>
                  <div className={`transition-all duration-500 absolute inset-x-0 space-y-4 ${activeFeature === 1 ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-4 scale-95 pointer-events-none'}`}>
                    <div className="text-6xl font-display text-white/5 select-none leading-none">02</div>
                    <h3 className="text-xl md:text-2xl font-bold text-white tracking-wide">Voice & Sleep Biometrics</h3>
                    <p className="text-sm font-light text-lime-200/60 leading-relaxed">Sometimes it’s hard to find the right words. AstroMind understands how you’re feeling under the surface.</p>
                  </div>
                  <div className={`transition-all duration-500 absolute inset-x-0 space-y-4 ${activeFeature === 2 ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-4 scale-95 pointer-events-none'}`}>
                    <div className="text-6xl font-display text-white/5 select-none leading-none">03</div>
                    <h3 className="text-xl md:text-2xl font-bold text-white tracking-wide">The Echoes Module</h3>
                    <p className="text-sm font-light text-lime-200/60 leading-relaxed">Distance shouldn't mean disconnection. AstroMind seamlessly connects you to 'Echoes'.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div ref={howItWorksContainer} className="relative h-[250vh] w-full bg-[#f4f7f0]">
            <div ref={howItWorksSticky} className="w-full h-screen flex items-center justify-center relative overflow-hidden">
              <div className="w-full h-full bg-[#f9fbf7] rounded-3xl relative shadow-2xl flex items-center justify-center overflow-hidden p-8 border border-neutral-200">
                <div ref={headingRef} className="absolute top-1/4 right-10 md:right-20 z-0 pointer-events-none text-right">
                  <h2 className="text-6xl md:text-[90px] font-black text-[#3c5e12] leading-[0.85] tracking-tight uppercase select-none">HOW ASTROMIND<br />WORKS</h2>
                </div>
                <div ref={centerMascotRef} className="absolute w-44 h-44 md:w-64 md:h-64 z-20 transition-transform duration-100">
                  <svg viewBox="0 0 100 100" className="w-full h-full drop-shadow-2xl">
                    <path fill="#eab308" d="M30,50 Q50,20 70,50 Q50,80 30,50 Z" /><circle cx="70" cy="50" r="8" fill="#1e293b" />
                  </svg>
                </div>
                <div className="w-full h-full relative z-10 flex items-center justify-center font-charon">
                  <div ref={cardPurpleRef} className="absolute left-6 md:left-16 bottom-24 w-[240px] md:w-[280px] bg-[#8b5cf6] text-white p-6 rounded-2xl shadow-xl flex flex-col justify-end">
                    <p className="text-base md:text-lg font-bold tracking-tight">Analyzes your voice during daily check-ins to instantly detect underlying markers of cognitive fatigue or psychological risk.</p>
                  </div>
                  <div ref={cardPinkRef} className="absolute right-6 md:right-16 bottom-12 w-[240px] md:w-[280px] bg-[#f0abfc] text-[#4a044e] p-6 rounded-2xl shadow-xl flex flex-col justify-end">
                    <p className="text-sm md:text-base font-bold leading-tight">Dad messages from Earth, targeted emotional anchors to combat the compounding effects of long-duration isolation.</p>
                  </div>
                  <div ref={cardOrangeRef} className="absolute w-[240px] md:w-[280px] bg-[#fb923c] text-[#7c2d12] p-6 rounded-2xl shadow-xl flex flex-col justify-end opacity-0">
                    <p className="text-sm md:text-base font-bold leading-tight">Processes physiological telemetry data from your onboard wearables, tracking your REM, deep sleep cycles, and resting heart rate.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* TRUST & PRIVACY BANNER */}
          <footer className=" bg-transparent text-[#f4fce3] px-5 pt-10 pb-6 font-mono selection:bg-[#f4fce3] selection:text-[#2d4206]">
            <div className="max-w-6xl mx-auto">
              <div className="flex flex-row items-center justify-center gap-6 mb-8 max-[768px]:flex-col max-[768px]:items-start max-[768px]:gap-3">
                <h1 className="text-[clamp(4rem,10vw,7.5rem)] font-black leading-[0.85] tracking-tighter uppercase m-0 selection:bg-[#a9ba8a]">
                  ASTROMIND
                </h1>
              </div>
              <hr className="border-t-2 border-dashed border-[#a9ba8a]/40 mb-5" />
              <div className="grid grid-cols-1 md:grid-cols-3 items-center gap-4 text-[11px] font-bold tracking-wider font-sans">
                <div className="text-center md:text-left">MADE WITH STARS.</div>
                <div className="flex gap-3 justify-center order-first md:order-none">
                  <a href="https://twitter.com" target="_blank" rel="noreferrer" className="bg-[#f4fce3] text-[#2d4206] w-7 h-7 rounded-full flex items-center justify-center hover:scale-110 hover:opacity-90 transition-transform duration-200">
                    <svg viewBox="0 0 24 24" className="w-[18px] h-[18px]" fill="currentColor">
                      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                    </svg>
                  </a>
                </div>
                <div className="text-center md:text-right">
                  <a href="#privacy" className="hover:underline">Give a star on GitHub</a>
                </div>
              </div>
            </div>
          </footer>

        </div>
      )}
    </div>
  );
};

export default AstroMindHome;