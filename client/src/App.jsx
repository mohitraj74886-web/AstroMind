import { useRef, useState, useEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger);

const AstroMindHome = () => {
  const videoRef = useRef(null);

  // Feature Section Refs
  const containerRef = useRef(null);
  const stickyRef = useRef(null);
  const fillBarRef = useRef(null);
  const glowDotRef = useRef(null);
  const [activeFeature, setActiveFeature] = useState(0);
  const [isYoYoMode, setIsYoYoMode] = useState(true);

  // 🚀 How It Works Section Refs
  const howItWorksContainer = useRef(null);
  const howItWorksSticky = useRef(null);
  const centerMascotRef = useRef(null);
  const cardPurpleRef = useRef(null);
  const cardPinkRef = useRef(null);
  const cardOrangeRef = useRef(null);
  const headingRef = useRef(null);

  // 1. Cross-browser Forward & Backward Video Loop Engine
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

  // 2. Unified GSAP Scroll Timeline (Features)
  useGSAP(() => {
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
  }, { scope: containerRef });


  // 3. 🚀 New "How It Works" Scroll Animation (2-Step Sequence)
  useGSAP(() => {
    // Pin layout canvas - increased height to 350vh to give the 2 steps breathing room
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

    // ==========================================
    // STEP 1: Purple exits, Pink shifts right, Orange arrives
    // ==========================================
    workTl.to(centerMascotRef.current, {
      rotation: 90,
      scale: 1.2,
      y: -20,
      ease: "power1.inOut"
    }, 0)
      .to(headingRef.current, {
        y: -40,
        scale: 0.9,
        opacity: 0.4,
        ease: "power1.inOut"
      }, 0)
      .to(cardPurpleRef.current, {
        x: -100,
        y: 150,
        opacity: 0,
        scale: 0.7,
        ease: "power1.inOut"
      }, 0)
      .fromTo(cardPinkRef.current,
        { x: 0, y: 180, scale: 0.8, opacity: 0.3 }, // Base layout coordinates via CSS
        { x: 180, y: -80, scale: 1, ease: "power1.inOut" },
        0
      )
      .fromTo(cardOrangeRef.current,
        { y: 300, x: -150, opacity: 0, scale: 0.5 },
        { y: 60, x: -160, opacity: 1, scale: 1, ease: "power1.inOut" },
        0
      );

    // ==========================================
    // STEP 2: Pink Card moves from right to center stage
    // ==========================================
    // This happens sequentially after step 1 finishes (default duration is 0.5s)
    workTl.to(cardPinkRef.current, {
      x: -80,            // Reset X translates to head to center
      y: -40,          // Fine-tune vertical center alignment
      scale: 1.1,      // Give it a focal emphasis pop
      zIndex: 50,      // Ensure it floats over elements
      ease: "power2.inOut",
      duration: 0.5,
      opacity: 1   // Dedicates distinct scrolling track time to this action
    })
      // Simultaneously fade out/scale back surrounding elements to highlight Pink Card
      .to(cardOrangeRef.current, {
        opacity: 0.3,
        scale: 0.9,
        ease: "power2.inOut",
        duration: 0.5
      }, "<")
      .to(centerMascotRef.current, {
        opacity: 1,
        scale: 0.8,
        rotate: 180,
        ease: "power2.inOut",
        duration: 0.5
      }, "<");

  }, { scope: howItWorksContainer });


  return (
    <div className="relative min-h-screen bg-black text-lime-100 overflow-x-hidden font-cursive">

      {/* BACKGROUND VIDEO LAYER */}
      <div className="fixed inset-0 w-full h-full z-0 overflow-hidden pointer-events-none">
        <video
          ref={videoRef}
          className="absolute top-1/2 left-1/2 w-full h-full object-cover -translate-x-1/2 -translate-y-1/2 opacity-25 mix-blend-overlay"
          src="./bg.mp4"
          muted
          playsInline
          autoPlay
        />
        <div className="absolute" />
      </div>

      {/* WEB INTERFACE PRESENTATION */}
      <div className="relative z-10 w-full">

        {/* HERO SECTION */}
        <header className="relative min-h-screen flex flex-col justify-center items-center px-6 text-center max-w-5xl mx-auto font-handwriting">
          <div className="space-y-6">
            <span className="inline-block text-xs md:text-xs tracking-[0.3em] uppercase text-lime-400 font-bold opacity-80">
              System Initialization
            </span>
            <h2 className="text-3xl md:text-7xl font-display uppercase tracking-tight text-white leading-none">
              Your quiet space <br />
              <span className="text-lime-300">among the stars.</span>
            </h2>
            <p className="text-base md:text-xl font-light text-lime-100/80 max-w-2xl mx-auto leading-relaxed font-charon">
              Deep space is vast, but you never have to navigate it alone. AstroMind is your private, onboard companion—here to listen, understand, and support your well-being.
            </p>
            <div className="pt-6 flex flex-col sm:flex-row gap-4 justify-center items-center">
              <button className="w-full sm:w-auto px-8 py-4 bg-lime-400 text-[#1b3002] rounded-full font-bold shadow-lg hover:bg-lime-300 transition-all duration-300">
                Begin Check-In
              </button>
              <button className="w-full sm:w-auto px-8 py-4 bg-black/40 text-lime-200 border border-lime-800/80 backdrop-blur-md rounded-full font-medium hover:bg-black/60 transition-all duration-300">
                Listen to Echoes
              </button>
            </div>
          </div>
        </header>

        {/* INTRODUCTION BRIDGE */}
        <section className="py-24 px-6 max-w-4xl mx-auto text-center font-charon">
          <h2 className="text-3xl md:text-5xl font-display uppercase text-white tracking-wide mb-6">
            A sanctuary for your mind.
          </h2>
          <p className="text-base md:text-lg font-light text-lime-100/70 leading-relaxed max-w-2xl mx-auto">
            Living in isolation requires immense resilience. AstroMind was built to be a steady anchor during your mission. By combining empathetic conversation with a gentle understanding of your body’s natural rhythms, we provide a safe, confidential space for you to process your thoughts, reflect on your days, and find a moment of peace.
          </p>
        </section>


        {/* ORIGINAL FEATURES CONTAINER */}
        <div ref={containerRef} className="relative h-[300vh] w-full font-charon">
          <div ref={stickyRef} className="w-full h-screen flex items-center justify-center bg-black/5 overflow-hidden">
            <div className="max-w-6xl w-full mx-auto px-6 grid grid-cols-1 md:grid-cols-12 items-center gap-8 relative h-full py-12">
              <div className="md:col-span-5 space-y-4 flex flex-col justify-center">
                <h2 className="text-5xl md:text-7xl font-display uppercase text-white leading-none tracking-tight">
                  What is <br />
                  <span className="text-lime-300">AstroMind?</span>
                </h2>
              </div>

              <div className="md:col-span-2 relative h-[350px] flex items-center justify-center">
                <div className="absolute inset-y-0 w-[2px] bg-white/10 rounded-full">
                  <div ref={fillBarRef} className="w-full bg-lime-400 origin-bottom absolute bottom-0 left-0 shadow-[0_0_10px_#a3e635]" style={{ height: "0%" }} />
                  <div ref={glowDotRef} className="absolute left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-slate-200 ring-4 ring-lime-500 shadow-[0_0_20px_#a3e635] z-10" style={{ bottom: "0%" }} />
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


        {/* 🚀 NEW SECTION: HOW IT WORKS (MATCHING SCREENSHOT 1 & 2) */}
        <div ref={howItWorksContainer} className="relative h-[250vh] w-full bg-[#f4f7f0]">
          <div ref={howItWorksSticky} className="w-full h-screen flex items-center justify-center relative overflow-hidden">

            {/* Main Stage Frame */}
            <div className="w-full h-full bg-[#f9fbf7] rounded-3xl relative shadow-2xl flex items-center justify-center overflow-hidden p-8 border border-neutral-200">

              {/* STICKY TRACKING HEADER TEXT */}
              <div ref={headingRef} className="absolute top-1/4 right-10 md:right-20 z-0 pointer-events-none text-right">
                <h2 className="text-6xl md:text-[90px] font-black text-[#3c5e12] leading-[0.85] tracking-tight uppercase select-none">
                  HOW ASTROMIND<br />WORKS
                </h2>
              </div>

              {/* CENTRAL MASCOT ELEMENT (Rotates between vertical/horizontal like the bee image) */}
              <div
                ref={centerMascotRef}
                className="absolute w-44 h-44 md:w-64 md:h-64 z-20 transition-transform duration-100"
                style={{ transform: "rotate(0deg)" }}
              >
                {/* Clean placeholder vector illustration styled like the golden bee */}
                <svg viewBox="0 0 100 100" className="w-full h-full drop-shadow-2xl">
                  <path fill="#eab308" d="M30,50 Q50,20 70,50 Q50,80 30,50 Z" />
                  <circle cx="70" cy="50" r="8" fill="#1e293b" />
                  <path d="M25,35 Q10,10 35,25 Z" fill="#93c5fd" opacity="0.8" />
                  <path d="M25,65 Q10,90 35,75 Z" fill="#93c5fd" opacity="0.8" />
                  <line x1="40" y1="38" x2="40" y2="62" stroke="#1e293b" strokeWidth="4" />
                  <line x1="50" y1="35" x2="50" y2="65" stroke="#1e293b" strokeWidth="4" />
                </svg>
              </div>

              {/* COMPONENT CARDS WRAPPER GRID */}
              <div className="w-full h-full relative z-10 flex items-center justify-center font-charon">

                {/* Card 1: Purple Block (Starts left, moves out) */}
                <div
                  ref={cardPurpleRef}
                  className="absolute left-6 md:left-16 bottom-24 w-[240px] md:w-[280px] bg-[#8b5cf6] text-white p-6 rounded-2xl shadow-xl flex flex-col justify-end"
                >
                  <div className="text-3xl font-bold mb-1">🌱</div>
                  <div className="text-4xl md:text-5xl font-black mb-2 flex items-baseline">
                    3<span className="text-lg md:text-xl font-normal ml-0.5">%</span>
                  </div>
                  <p className="text-base md:text-lg font-bold tracking-tight">Buys Farmland.</p>
                </div>

                {/* Card 2: Pink/Lavender Block (Starts tucked bottom right, scales into placement) */}
                <div
                  ref={cardPinkRef}
                  className="absolute right-6 md:right-16 bottom-12 w-[240px] md:w-[280px] bg-[#f0abfc] text-[#4a044e] p-6 rounded-2xl shadow-xl flex flex-col justify-end"
                >
                  <div className="text-3xl mb-1">🚜</div>
                  <div className="text-4xl md:text-5xl font-black mb-2 flex items-baseline">
                    1<span className="text-lg md:text-xl font-normal ml-0.5">%</span>
                  </div>
                  <p className="text-sm md:text-base font-bold leading-tight">Pays For Seeds, Tractors, Farmers.</p>
                </div>

                {/* Card 3: Orange Block (Hidden initially, slides into left region) */}
                <div
                  ref={cardOrangeRef}
                  className="absolute w-[240px] md:w-[280px] bg-[#fb923c] text-[#7c2d12] p-6 rounded-2xl shadow-xl flex flex-col justify-end opacity-0"
                >
                  <div className="text-3xl mb-1">💻</div>
                  <div className="text-4xl md:text-5xl font-black mb-2 flex items-baseline">
                    1<span className="text-lg md:text-xl font-normal ml-0.5">%</span>
                  </div>
                  <p className="text-sm md:text-base font-bold leading-tight">Helps Grow Our Community & Tech.</p>
                </div>

              </div>

            </div>
          </div>
        </div>


        {/* TRUST & PRIVACY BANNER */}
        <footer className=" bg-transparent text-[#f4fce3] px-5 pt-10 pb-6 font-mono selection:bg-[#f4fce3] selection:text-[#2d4206]">
          <div className="max-w-6xl mx-auto">

            {/* Main Branding Section */}
            <div className="flex flex-row items-center justify-center gap-6 mb-8 max-[768px]:flex-col max-[768px]:items-start max-[768px]:gap-3">
              <h1 className="text-[clamp(4rem,10vw,7.5rem)] font-black leading-[0.85] tracking-tighter uppercase m-0 selection:bg-[#a9ba8a]">
                ASTROMIND
              </h1>
            </div>

            {/* Dashed Divider Line */}
            <hr className="border-t-2 border-dashed border-[#a9ba8a]/40 mb-5" />

            {/* Bottom Bar Section */}
            <div className="grid grid-cols-1 md:grid-cols-3 items-center gap-4 text-[11px] font-bold tracking-wider font-sans">

              {/* Left Tagline */}
              <div className="text-center md:text-left">
                MADE WITH STARS.
              </div>

              {/* Center Social Icons */}
              <div className="flex gap-3 justify-center order-first md:order-none">
                {/* Twitter / X */}
                <a
                  href="https://twitter.com"
                  target="_blank"
                  rel="noreferrer"
                  className="bg-[#f4fce3] text-[#2d4206] w-7 h-7 rounded-full flex items-center justify-center hover:scale-110 hover:opacity-90 transition-transform duration-200"
                >
                  <svg viewBox="0 0 24 24" className="w-[18px] h-[18px]" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                </a>
              </div>

              {/* Right Links */}
              <div className="text-center md:text-right">
                <a href="#privacy" className="hover:underline">Give a star on GitHub</a>
              </div>

            </div>
          </div>
        </footer>


      </div>
    </div>
  );
};

export default AstroMindHome;
