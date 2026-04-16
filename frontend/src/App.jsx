import { useState, useRef, useEffect, useMemo } from 'react';
import axios from 'axios';
import {
    Mic, Square, Activity, AlertTriangle, CheckCircle,
    Loader2, Info, Server, Phone, Mail, User, Calendar, Shield,
    FileText, Download, Sun, Moon, ArrowRight, Upload, RefreshCcw,
    BrainCircuit, Headphones, ChevronLeft,
    ThumbsUp, ThumbsDown, ShieldCheck
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Helper for Session ID
const generateSessionId = () => crypto.randomUUID();

const API_BASE = (import.meta.env.VITE_API_URL || window.location.origin + '/api').replace(/\/$/, '');
const WS_BASE = API_BASE.replace(/^http/, 'ws');

const STATES = {
    IDLE: 'IDLE',           // Home / Selection
    INFO: 'INFO',           // Step 1: Input Info
    RECORDING: 'RECORDING', // Step 2: Recording
    VALIDATING: 'VALIDATING', // Step 3: Checking Validity (Auto)
    REVIEW_VALID: 'REVIEW_VALID', // Step 3: Valid -> Ready to Analyze
    REVIEW_INVALID: 'REVIEW_INVALID', // Step 3: Invalid -> Prompt Retry
    ANALYZING: 'ANALYZING', // Step 4: Calls Backend
    RESULT: 'RESULT',       // Step 5: Success
    ERROR: 'ERROR'          // Step 5: Failure
};

function App() {
    const [theme, setTheme] = useState('light');
    const [page, setPage] = useState('home'); // 'home' | 'screening' | 'tech' | 'guide'

    // Navigation Helper
    const handleNav = (targetPage) => {
        // Stop any active recording before navigating away
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        if (audioContextRef.current) {
            audioContextRef.current.close().catch(() => { });
        }
        setPage(targetPage);
        // appState will be set by the useEffect below based on the target page
    };

    // --- STATE MACHINE ---
    const [appState, setAppState] = useState(STATES.IDLE);

    // Patient Data
    const [patientInfo, setPatientInfo] = useState({ name: '', dob: '', age: '', gender: 'Nam', health_notes: '', dialect: 'North' });

    // Session Data
    const sessionIdRef = useRef(generateSessionId());

    // Audio Data
    const [audioBlob, setAudioBlob] = useState(null);
    const [validationData, setValidationData] = useState(null); // Full Quality Metrics
    const [analysisData, setAnalysisData] = useState(null);     // Full Result
    const [historyData, setHistoryData] = useState([]);         // Trend Data
    const [realtimeSai, setRealtimeSai] = useState(null);       // Real-time AI Insight
    const [realtimeMetrics, setRealtimeMetrics] = useState({}); // Live Biomarkers
    const [error, setError] = useState(null);

    // UI Helpers
    const [audioVisual, setAudioVisual] = useState(new Array(10).fill(0));
    const [recordingTime, setRecordingTime] = useState(0);
    const MAX_RECORDING_TIME = 20;

    // VAD State
    const [vadStatus, setVadStatus] = useState('SILENCE');
    const [displayValidTime, setDisplayValidTime] = useState(0);
    const validDurationRef = useRef(0);
    const speechDetectedRef = useRef(false);

    // Refs
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const animationFrameRef = useRef(null);
    const timerRef = useRef(null);
    const silenceStartRef = useRef(null);
    const socketRef = useRef(null);

    // --- LOGIC ---

    const processAudio = () => {
        if (!analyserRef.current) return;
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Smooth visualization
        const bars = 15;
        const step = Math.floor(dataArray.length / bars);
        const newVisual = [];
        for (let i = 0; i < bars; i++) {
            let sum = 0;
            for (let j = 0; j < step; j++) sum += dataArray[i * step + j];
            newVisual.push(sum / step / 255);
        }
        setAudioVisual(newVisual);

        // VAD Logic
        const volume = dataArray.reduce((src, val) => src + val, 0) / dataArray.length;
        const threshold = 15;

        if (volume > threshold) {
            setVadStatus('SPEAKING');
            if (!speechDetectedRef.current) speechDetectedRef.current = true;
            validDurationRef.current += 0.016;
            setDisplayValidTime(validDurationRef.current);
            silenceStartRef.current = null;
        } else {
            setVadStatus('SILENCE');
        }

        // Send to WebSocket if open
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            // Processing audio frames for VAD is done, but streaming is handled in mediaRecorder.ondataavailable
        }

        animationFrameRef.current = requestAnimationFrame(processAudio);
    };

    const fetchHistory = async () => {
        try {
            const res = await axios.get(`${API_BASE}/history`);
            setHistoryData(res.data);
        } catch (e) {
            console.error("Failed to fetch history", e);
        }
    };


    const startRecording = async () => {
        try {
            if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                mediaRecorderRef.current.stop();
            }
            if (audioContextRef.current) audioContextRef.current.close().catch(() => { });
            if (timerRef.current) clearInterval(timerRef.current);
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

            // STATE TRANSITION
            setAppState(STATES.RECORDING);
            setValidationData(null);
            setAnalysisData(null);
            setError(null);
            setAudioBlob(null);
            setRealtimeSai(null);
            setRealtimeMetrics({});

            // WebSocket Setup - Real-time Voice Stream
            try {
                const wsUrl = `${WS_BASE}/ws/voice-stream?gender=${patientInfo.gender || 'Nam'}`;
                socketRef.current = new WebSocket(wsUrl);
                socketRef.current.onmessage = (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        if (data.type === 'STREAM_UPDATE') {
                            setRealtimeSai(data.sai_score);
                            setRealtimeMetrics(data.metrics || {});
                        }
                    } catch (_) { /* ignore parse errors */ }
                };
                socketRef.current.onerror = () => {
                    socketRef.current = null; // graceful degradation
                };
            } catch (_) {
                socketRef.current = null;
            }

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Resume AudioContext (required by many browsers after user gesture)
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            if (audioContextRef.current.state === 'suspended') {
                await audioContextRef.current.resume();
            }

            // MediaRecorder Setup with fallback mimeTypes
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorderRef.current.onstop = async () => {
                if (audioChunksRef.current.length === 0) {
                    console.error("No audio data captured.");
                    setError("Không thu được dữ liệu âm thanh. Vui lòng thử lại.");
                    setAppState(STATES.INFO);
                    return;
                }
                const blob = new Blob(audioChunksRef.current, { type: mimeType });
                setAudioBlob(blob);

                // STATE TRANSITION -> VALIDATING
                setAppState(STATES.VALIDATING);
                checkAudioQuality(blob);

                stream.getTracks().forEach(t => t.stop());
                if (audioContextRef.current) {
                    audioContextRef.current.close().catch(e => console.error(e));
                    audioContextRef.current = null;
                }
            };

            mediaRecorderRef.current.start(500);
            setRecordingTime(0);

            // Stats Reset
            validDurationRef.current = 0;
            setDisplayValidTime(0);
            setVadStatus('SILENCE');

            // Timer
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => {
                    if (prev >= MAX_RECORDING_TIME) {
                        stopRecording();
                        return MAX_RECORDING_TIME;
                    }
                    return prev + 1;
                });
            }, 1000);

            // VAD & RAW PCM Streaming Setup
            const source = audioContextRef.current.createMediaStreamSource(stream);
            const analyser = audioContextRef.current.createAnalyser();
            analyser.fftSize = 256;
            analyserRef.current = analyser; // Set the ref for processAudio

            // ScriptProcessor for RAW PCM
            const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                }

                if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
                    socketRef.current.send(pcmData.buffer);
                }
            };

            source.connect(analyser);
            source.connect(processor);
            processor.connect(audioContextRef.current.destination);

            // Force a slight delay to ensure nodes are pulling data
            setTimeout(() => {
                processAudio();
            }, 100);

        } catch (err) {
            console.error(err);
            setError("Không thể truy cập Microphone. Vui lòng cấp quyền.");
            setAppState(STATES.ERROR);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        if (socketRef.current) {
            socketRef.current.close();
            socketRef.current = null;
        }
        if (timerRef.current) clearInterval(timerRef.current);
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

        // Ensure AudioContext is closed to stop processor
        if (audioContextRef.current) {
            audioContextRef.current.close().catch(() => { });
            audioContextRef.current = null;
        }
    };

    const checkAudioQuality = async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'check.wav');

        try {
            const res = await axios.post(`${API_BASE}/validate`, formData);
            if (res.data && res.data.status === 'VALID') {
                setValidationData(res.data.quality_metrics);
                setAppState(STATES.REVIEW_VALID);
            } else {
                setValidationData(res.data.quality_metrics || {});
                setAppState(STATES.REVIEW_INVALID);
                setError(res.data.reasons ? res.data.reasons.join(', ') : "Chất lượng không đạt");
            }
        } catch (err) {
            console.error("Validation Error", err);
            setAppState(STATES.ERROR);
            setError("Lỗi kết nối kiểm tra âm thanh.");
        }
    };

    const handleAnalyze = async () => {
        if (!audioBlob) return;
        if (audioBlob.size < 1000) {
            setError("File âm thanh quá nhỏ hoặc lỗi.");
            return;
        }

        if (appState !== STATES.REVIEW_VALID) {
            setError("Bạn cần nói vào micro trước khi phân tích.");
            return;
        }

        setAppState(STATES.ANALYZING);
        setError(null);

        const formData = new FormData();
        formData.append('file', audioBlob, 'voice.wav');
        formData.append('name', patientInfo.name);
        formData.append('dob', patientInfo.dob);
        formData.append('age', patientInfo.age);
        formData.append('gender', patientInfo.gender);
        formData.append('health_notes', patientInfo.health_notes);
        formData.append('validation_status', validationData ? 'VALID' : 'INVALID');
        formData.append('dialect', patientInfo.dialect || 'North');

        const config = {
            headers: {
                'X-Request-ID': sessionIdRef.current
            }
        };

        try {
            const res = await axios.post(`${API_BASE}/analyze`, formData, config);

            if (res.data && res.data.status === 'SUCCESS') {
                setAnalysisData(res.data);
                setAppState(STATES.RESULT);
            } else if (res.data && (res.data.status === 'INPUT_REJECTED' || res.data.status === 'INVALID_AUDIO_INPUT')) {
                setAppState(STATES.REVIEW_INVALID);
                setError(res.data.explanation || "Dữ liệu bị từ chối.");
            } else {
                setAppState(STATES.ERROR);
                setError(res.data.message || "Phân tích thất bại.");
            }
        } catch (err) {
            console.error(err);
            setAppState(STATES.ERROR);
            setError("Lỗi kết nối Server.");
        }
    };

    const handleConsultExpert = async () => {
        try {
            const formData = new FormData();
            formData.append('session_id', analysisData?.session_id);
            await axios.post(`${API_BASE}/consult`, formData);
            alert("Đã gửi yêu cầu tư vấn đến chuyên gia. Chúng tôi sẽ phản hồi sớm nhất qua email/SĐT của bạn.");
        } catch (e) {
            console.error("Consultation failed", e);
            alert("Lỗi khi gửi yêu cầu tư vấn.");
        }
    };

    const handleFeedback = async (rating) => {
        try {
            await axios.post(`${API_BASE}/feedback`, {
                session_id: analysisData?.session_id,
                rating: rating,
                risk_assessment: analysisData?.final_risk_level,
                confidence: analysisData?.confidence_score
            });
            alert("Cảm ơn bạn đã phản hồi!");
        } catch (e) {
            console.error("Feedback failed", e);
        }
    };

    const downloadReport = () => {
        if (!analysisData || !analysisData.report_url) return;
        window.open(`${API_BASE}${analysisData.report_url}`, '_blank');
    };

    const downloadZip = () => {
        if (!analysisData || !analysisData.report_zip_url) return;
        window.open(`${API_BASE}${analysisData.report_zip_url}`, '_blank');
    };

    useEffect(() => {
        document.documentElement.classList.toggle('dark', theme === 'dark');
    }, [theme]);

    // Reset Logic when page changes
    useEffect(() => {
        if (page === 'home') {
            setAppState(STATES.IDLE);
            setPatientInfo({ name: '', dob: '', age: '', gender: 'Nam', health_notes: '', dialect: 'North' });
            setAudioBlob(null);
            setValidationData(null);
            setAnalysisData(null);
            setError(null);
            setRealtimeSai(null);
            setRealtimeMetrics({});
        } else if (page === 'screening') {
            // CRITICAL: Must clear previous analysis to avoid skip-to-results bug
            setAnalysisData(null);
            setAppState(STATES.INFO);
            setAudioBlob(null);
            setValidationData(null);
            setError(null);
            setRealtimeSai(null);
            setRealtimeMetrics({});
            setRecordingTime(0);
            setDisplayValidTime(0);
            validDurationRef.current = 0;
            sessionIdRef.current = generateSessionId();
        } else if (page === 'trend') {
            fetchHistory();
        }
    }, [page]);

    const resetSession = () => {
        setAudioBlob(null);
        setValidationData(null);
        setAnalysisData(null);
        setError(null);
        setRecordingTime(0);
        setDisplayValidTime(0);
        validDurationRef.current = 0;
        sessionIdRef.current = generateSessionId(); // New session ID
    };

    const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');



    // Components
    const PatientInfoCard = ({ info, metadata }) => {
        if (!info) return null;
        return (
            <div className="medical-card p-6 flex flex-wrap gap-8 items-center justify-between border-l-4 border-l-sky-500 shadow-sky-500/5 mb-6">
                <div className="flex gap-6 items-center">
                    <div className="p-3 bg-sky-500/10 rounded-2xl text-sky-600">
                        <User size={24} />
                    </div>
                    <div>
                        <h3 className="text-lg font-black text-[var(--text-main)] uppercase tracking-tight">{info.name || "Bệnh nhân"}</h3>
                        <div className="flex gap-4 mt-1">
                            <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-1">
                                <Calendar size={12} /> {info.dob || "N/A"} ({info.age || "N/A"} tuổi)
                            </span>
                            <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-1">
                                <Activity size={12} /> {info.gender || "Nam"}
                            </span>
                        </div>
                    </div>
                </div>

                <div className="flex flex-wrap gap-4">
                    <div className="px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-xl">
                        <span className="block text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1">Vùng miền</span>
                        <span className="text-xs font-bold text-sky-600">{metadata?.dialect || "Miền Bắc"}</span>
                    </div>
                    <div className="px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-xl">
                        <span className="block text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1">Thời điểm</span>
                        <span className="text-xs font-bold text-slate-600">{metadata?.timestamp ? new Date(metadata.timestamp).toLocaleString('vi-VN') : "Vừa xong"}</span>
                    </div>
                </div>
            </div>
        );
    };

    const SpectrumChart = ({ data }) => {
        return (
            <div className="h-64 mt-4 bg-white/50 dark:bg-slate-900/40 rounded-3xl p-6 border border-sky-500/10 shadow-inner">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-[10px] font-black uppercase text-sky-500 dark:text-sky-400 tracking-widest flex items-center gap-2">
                        <Activity size={14} /> Phân tích Phổ âm học (Spectral Analysis)
                    </h3>
                    <div className="flex gap-4 text-[8px] font-bold text-slate-400 dark:text-slate-500">
                        <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-sky-500" /> Tín hiệu tuyệt đối (Waveform)</span>
                    </div>
                </div>
                <div className="h-[80%] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data || []}>
                            <defs>
                                <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.1} />
                                </linearGradient>
                            </defs>
                            <XAxis dataKey="name" hide />
                            <YAxis hide domain={[0, 'dataMax']} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '12px', fontSize: '10px', color: '#fff' }}
                                itemStyle={{ color: '#0ea5e9' }}
                                labelStyle={{ display: 'none' }}
                                formatter={(val) => [val.toFixed(4), "Biên độ"]}
                            />
                            <Area type="monotone" dataKey="value" stroke="#0ea5e9" fillOpacity={1} fill="url(#colorVal)" strokeWidth={2} isAnimationActive={false} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        );
    };

    const ProgressBar = ({ value, label, min, max, unit }) => {
        const percentage = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
        const status = percentage < 40 ? "NORMAL" : percentage < 75 ? "MODERATE" : "HIGH";
        const statusColor = status === "NORMAL" ? "emerald" : status === "MODERATE" ? "amber" : "rose";

        return (
            <div className="space-y-2">
                <div className="flex justify-between items-end">
                    <div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{label}</span>
                        <div className="text-sm font-black text-[var(--text-main)]">{value.toFixed(1)} <span className="text-[10px] text-slate-500 font-normal">{unit}</span></div>
                    </div>
                    <span className={`text-[8px] font-black px-2 py-0.5 rounded-full bg-${statusColor}-500/10 text-${statusColor}-500 border border-${statusColor}-500/20`}>
                        {status}
                    </span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden flex">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${percentage}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className={`h-full bg-gradient-to-r ${status === 'NORMAL' ? 'from-emerald-500 to-emerald-400' : status === 'MODERATE' ? 'from-amber-500 to-amber-400' : 'from-rose-500 to-rose-400'} shadow-lg`}
                    />
                </div>
            </div>
        );
    };

    const WaveformDisplay = ({ seed = 0.5 }) => {
        // Generate a pseudo-random scientific waveform path
        const points = 100;
        const width = 800;
        const height = 80;
        const path = [];
        for (let i = 0; i < points; i++) {
            const x = (i / points) * width;
            // Create a medical "burst" pattern
            const amplitude = Math.sin(i * 0.2) * Math.cos(i * 0.05) * 30 * (0.5 + Math.random() * 0.5);
            const y = height / 2 + amplitude;
            path.push(`${i === 0 ? 'M' : 'L'} ${x} ${y}`);
        }

        return (
            <div className="relative w-full h-24 bg-[var(--card-muted)] rounded-2xl overflow-hidden border border-[var(--card-border)] flex items-center justify-center">
                <div className="absolute inset-0 opacity-10" style={{
                    backgroundImage: 'linear-gradient(var(--card-border) 1px, transparent 1px), linear-gradient(90deg, var(--card-border) 1px, transparent 1px)',
                    backgroundSize: '20px 20px'
                }} />
                <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full px-4 overflow-visible">
                    <defs>
                        <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#0ea5e9" />
                            <stop offset="50%" stopColor="#6366f1" />
                            <stop offset="100%" stopColor="#10b981" />
                        </linearGradient>
                    </defs>
                    <motion.path
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{ pathLength: 1, opacity: 1 }}
                        transition={{ duration: 2, ease: "easeInOut" }}
                        d={path.join(' ')}
                        fill="none"
                        stroke="url(#waveGradient)"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                    />
                    {/* Add markers for Biomarkers */}
                    {[20, 45, 75].map((pos, idx) => (
                        <g key={idx}>
                            <line x1={(pos / 100) * width} y1="0" x2={(pos / 100) * width} y2={height} stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="2 2" />
                            <circle cx={(pos / 100) * width} cy={height / 2} r="2" fill="#6366f1" />
                        </g>
                    ))}
                </svg>
                <div className="absolute top-2 right-4 flex gap-4 text-[8px] font-black uppercase tracking-widest text-slate-400">
                    <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-sky-500" /> Tần số</span>
                    <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> Biên độ</span>
                </div>
            </div>
        );
    };

    return (
        <div className={`min-h-screen scientific-bg transition-colors duration-500`}>
            {/* Header */}
            <nav className="fixed top-0 w-full z-50 bg-[var(--card-bg)]/80 backdrop-blur-xl border-b border-[var(--card-border)] px-6 py-4 flex justify-between items-center transition-colors">
                <div className="flex items-center gap-3 cursor-pointer" onClick={() => handleNav('home')}>
                    <div className="bg-sky-600 p-2 rounded-xl text-white shadow-lg shadow-sky-200">
                        <BrainCircuit size={28} />
                    </div>
                    <div>
                        <h1 className="text-xl font-black gradient-heading tracking-tight">AI STROKE DETECTION</h1>
                        <p className="text-[10px] font-bold text-sky-500 uppercase tracking-[0.2em]"></p>
                    </div>
                </div>

                <div className="flex items-center gap-6">
                    <div className="hidden md:flex gap-6 text-sm font-bold text-[var(--text-muted)]">
                        <button onClick={() => handleNav('home')} className={`hover:text-sky-500 transition-colors ${page === 'home' ? 'text-sky-600' : ''}`}>Trang chủ</button>
                        <button onClick={() => handleNav('trend')} className={`hover:text-sky-500 transition-colors ${page === 'trend' ? 'text-sky-600' : ''}`}>Xu hướng</button>
                        <button onClick={() => handleNav('tech')} className={`hover:text-sky-500 transition-colors ${page === 'tech' ? 'text-sky-600' : ''}`}>Công nghệ</button>
                        <button onClick={() => handleNav('guide')} className={`hover:text-sky-500 transition-colors ${page === 'guide' ? 'text-sky-600' : ''}`}>Hướng dẫn</button>
                    </div>

                    <button
                        onClick={toggleTheme}
                        className="p-3 bg-slate-100 dark:bg-slate-800 rounded-2xl hover:bg-slate-200 dark:hover:bg-slate-700 transition-all text-[var(--text-main)]"
                    >
                        {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
                    </button>
                </div>
            </nav>

            <main className="pt-32 pb-40 px-4 max-w-[1600px] mx-auto">
                <AnimatePresence mode="wait">
                    {/* PAGE: HOME */}
                    {page === 'home' && (
                        <motion.div
                            key="home"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center"
                        >
                            <div className="space-y-8">
                                <span className="px-4 py-2 bg-sky-100 dark:bg-sky-900/30 text-sky-600 dark:text-sky-400 rounded-full text-xs font-black uppercase tracking-widest">
                                    Công nghệ Hybrid AI 2.0
                                </span>
                                <h1 className="text-5xl md:text-6xl font-black text-[var(--text-main)] leading-[1.1]">
                                    Phát hiện sớm <br />
                                    <span className="text-sky-600 italic">Nguy cơ Đột quỵ</span> <br />
                                    Qua giọng nói
                                </h1>
                                <div className="text-lg text-[var(--text-muted)] leading-relaxed max-w-xl">
                                    Hệ thống AI hỗ trợ tầm soát tại nhà nhanh chóng, an toàn. Phù hợp cho:
                                    <ul className="list-disc pl-5 mt-2 space-y-1 text-sm font-semibold text-slate-500">
                                        <li>Người nghi ngờ có dấu hiệu rối loạn giọng nói</li>
                                        <li>Người sau đột quỵ cần theo dõi phục hồi</li>
                                        <li>Người muốn kiểm tra chất lượng phát âm định kỳ</li>
                                    </ul>
                                </div>
                                <div className="flex flex-wrap gap-4 pt-4">
                                    <button onClick={() => setPage('screening')} className="btn-primary group">
                                        Bắt đầu Kiểm tra <ArrowRight className="group-hover:translate-x-1 transition-transform" />
                                    </button>
                                </div>
                                <div className="grid grid-cols-3 gap-6 pt-8 border-t border-[var(--card-border)]">
                                    <div>
                                        <p className="text-2xl font-black text-sky-600">1.8s</p>
                                        <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider">Tốc độ xử lý</p>
                                    </div>
                                    <div>
                                        <p className="text-2xl font-black text-sky-600">54</p>
                                        <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider">Biomarkers</p>
                                    </div>
                                    <div>
                                        <p className="text-2xl font-black text-sky-600">Consensus</p>
                                        <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider">Hybrid Engine</p>
                                    </div>
                                </div>
                            </div>

                            <div className="relative">
                                <div className="absolute inset-0 bg-sky-500/20 blur-[100px] rounded-full" />
                                <motion.div
                                    animate={{ y: [0, -20, 0] }}
                                    transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
                                    className="medical-card relative z-10 border-sky-500/30 dark:border-sky-500/10"
                                >
                                    <div className="flex gap-2 mb-8 items-center border-b border-[var(--card-border)] pb-6">
                                        <div className="w-12 h-12 bg-sky-100 dark:bg-sky-900/30 rounded-xl flex items-center justify-center text-sky-600">
                                            <BrainCircuit />
                                        </div>
                                        <h3 className="font-black text-[var(--text-main)]">Quy trình chuyên nghiệp</h3>
                                    </div>
                                    <div className="space-y-6">
                                        {[
                                            { step: 1, title: 'Định danh đối tượng', desc: 'Nhập tuổi và giới tính để hiệu chuẩn AI' },
                                            { step: 2, title: 'Trắc nghiệm giọng nói', desc: 'Thu âm 5s để phân tích 54 biomarkers' },
                                            { step: 3, title: 'Nhận báo cáo Y khoa', desc: ' AI đưa ra kết luận ' }
                                        ].map(item => (
                                            <div key={item.step} className="flex gap-4 items-start">
                                                <div className="step-indicator bg-sky-100 dark:bg-sky-900/40 text-sky-600">{item.step}</div>
                                                <div>
                                                    <h4 className="font-bold text-[var(--text-main)]">{item.title}</h4>
                                                    <p className="text-sm text-[var(--text-muted)]">{item.desc}</p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </motion.div>
                            </div>
                        </motion.div>
                    )}

                    {/* PAGE: TREND */}
                    {page === 'trend' && (
                        <motion.div key="trend" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-4xl mx-auto space-y-8">
                            <div className="flex justify-between items-center">
                                <h1 className="text-4xl font-black gradient-heading">Xu hướng Sức khỏe</h1>
                                <button onClick={() => setPage('screening')} className="btn-primary py-2 px-4 text-xs">Phân tích Mới</button>
                            </div>

                            {historyData.length > 0 ? (
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                    <div className="md:col-span-2 medical-card p-8">
                                        <h3 className="font-bold mb-6 flex items-center gap-2">
                                            <Activity size={18} className="text-sky-500" /> Biến thiên chỉ số SAI
                                        </h3>
                                        <div className="h-64 flex items-end gap-3 pb-8 border-b border-slate-100 dark:border-slate-800">
                                            {historyData.slice(-10).map((item, idx) => (
                                                <motion.div
                                                    key={idx}
                                                    initial={{ height: 0 }}
                                                    animate={{ height: `${Math.max(10, item.sai_score || 20)}%` }}
                                                    className="flex-1 bg-sky-500/20 hover:bg-sky-500/40 rounded-t-lg relative group transition-colors"
                                                >
                                                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 text-[10px] font-bold opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap bg-slate-800 text-white px-2 py-1 rounded">
                                                        SAI: {(item.sai_score || 0).toFixed(1)}
                                                    </div>
                                                </motion.div>
                                            ))}
                                        </div>
                                        <div className="flex justify-between mt-4 text-[10px] font-black text-slate-400 uppercase tracking-widest">
                                            <span>Quá khứ</span>
                                            <span>Hiện tại</span>
                                        </div>
                                    </div>

                                    <div className="space-y-6">
                                        <div className="medical-card p-6 border-sky-100">
                                            <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Trung bình SAI</p>
                                            <p className="text-3xl font-black text-sky-600">
                                                {(historyData.reduce((acc, cur) => acc + (cur.sai_score || 0), 0) / historyData.length).toFixed(1)}
                                            </p>
                                        </div>
                                        <div className="medical-card p-6">
                                            <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Số lần tầm soát</p>
                                            <p className="text-3xl font-black text-slate-600">{historyData.length}</p>
                                        </div>
                                    </div>

                                    <div className="md:col-span-3 space-y-4">
                                        <h4 className="font-bold text-sm text-slate-500">Lịch sử chi tiết</h4>
                                        {historyData.reverse().map((item, idx) => (
                                            <div key={idx} className="flex items-center justify-between p-4 bg-white dark:bg-slate-900 rounded-xl border border-slate-100 dark:border-slate-800 hover:border-sky-200 transition-all">
                                                <div className="flex gap-4 items-center">
                                                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold ${(item.sai_score || 0) < 30 ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>
                                                        SAI
                                                    </div>
                                                    <div>
                                                        <p className="text-sm font-bold text-[var(--text-main)]">Phiên phân tích {idx + 1}</p>
                                                        <p className="text-[10px] text-slate-400">{item.timestamp}</p>
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <p className="font-black text-slate-700">{(item.sai_score || 0).toFixed(1)}</p>
                                                    <p className="text-[10px] font-bold text-slate-400 uppercase">Score</p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                <div className="medical-card py-20 text-center space-y-4">
                                    <Activity className="mx-auto text-slate-300" size={48} />
                                    <p className="text-slate-500 font-bold">Chưa có dữ liệu lịch sử. Hãy thực hiện tầm soát đầu tiên!</p>
                                    <button onClick={() => setPage('screening')} className="btn-outline mx-auto">Bắt đầu ngay</button>
                                </div>
                            )}
                            <button onClick={() => handleNav('home')} className="btn-outline mx-auto block mt-12">Quay lại Trang chủ</button>
                        </motion.div>
                    )}

                    {/* PAGE: TECHNOLOGY */}
                    {page === 'tech' && (
                        <motion.div key="tech" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-4xl mx-auto space-y-8">
                            <h2 className="text-4xl font-black gradient-heading">Công nghệ Hybrid AI</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="medical-card">
                                    <BrainCircuit className="text-sky-600 mb-4" size={32} />
                                    <h3 className="text-xl font-bold mb-2 text-[var(--text-main)]">Consensus Engine</h3>
                                    <p className="text-[var(--text-muted)] text-sm leading-relaxed">
                                        Sử dụng sự kết hợp giữa mô hình học máy có giám sát (Random Forest) và
                                        mô hình không giám sát (Isolation Forest) để xác định các dị biệt trong chuỗi giọng nói micro-vocal.
                                    </p>
                                </div>
                                <div className="medical-card">
                                    <Activity className="text-sky-600 mb-4" size={32} />
                                    <h3 className="text-xl font-bold mb-2 text-[var(--text-main)]">54 Bio-markers</h3>
                                    <p className="text-[var(--text-muted)] text-sm leading-relaxed">
                                        Bóc tách 54 đặc trưng âm học bao gồm Voice Onset Time (VOT), Formant Stability,
                                        Shimmer, Jitter và các biến thiên nhịp điệu để phát hiện dấu hiệu liệt cơ nhẹ.
                                    </p>
                                </div>
                            </div>
                            <button onClick={() => handleNav('home')} className="btn-outline mx-auto block mt-12">Quay lại Trang chủ</button>
                        </motion.div>
                    )}

                    {/* PAGE: GUIDE */}
                    {page === 'guide' && (
                        <motion.div key="guide" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-3xl mx-auto space-y-12">
                            <div className="text-center">
                                <h2 className="text-4xl font-black mb-4 text-[var(--text-main)]">Hướng dẫn trắc nghiệm</h2>
                                <p className="text-[var(--text-muted)]">Cần tuân thủ 3 yêu cầu sau để có kết quả chính xác nhất</p>
                            </div>
                            <div className="space-y-6">
                                {[
                                    { icon: <Headphones />, title: "Môi trường yên tĩnh", text: "Tránh tiếng ồn từ quạt, tivi hoặc người nói xung quanh." },
                                    { icon: <Mic />, title: "Khoảng cách Micro", text: "Đặt điện thoại/micro cách miệng khoảng 15-20cm, không thổi hơi trực tiếp." },
                                    { icon: <FileText />, title: "Nội dung trắc nghiệm", text: "Hãy đọc văn bản được yêu cầu một cách tự nhiên, không cố diễn cảm." }
                                ].map((step, idx) => (
                                    <div key={idx} className="medical-card flex items-center gap-6 p-8">
                                        <div className="text-sky-600">{step.icon}</div>
                                        <div>
                                            <h3 className="font-bold text-lg text-[var(--text-main)]">{step.title}</h3>
                                            <p className="text-[var(--text-muted)] text-sm">{step.text}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <button onClick={() => handleNav('home')} className="btn-outline mx-auto block">Tôi đã hiểu</button>
                        </motion.div>
                    )}


                    {/* PAGE: SCREENING */}
                    {page === 'screening' && (
                        <motion.div
                            key="screening"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 1.05 }}
                            className="max-w-2xl mx-auto"
                        >
                            {appState !== STATES.RESULT && (
                                <button
                                    onClick={() => setPage('home')}
                                    className="mb-8 flex items-center gap-2 text-sm font-bold text-sky-600 hover:gap-3 transition-all"
                                >
                                    <ChevronLeft size={18} /> Quay lại
                                </button>
                            )}

                            {/* STATE: INFO */}
                            {appState === STATES.INFO && (
                                <div className="space-y-8">
                                    <div className="medical-card">
                                        <h2 className="text-2xl font-black mb-6 text-[var(--text-main)]">1. Nhập thông tin xác thực</h2>
                                        <div className="space-y-5">
                                            <div>
                                                <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Tên</label>
                                                <input
                                                    type="text"
                                                    value={patientInfo.name}
                                                    onChange={e => setPatientInfo({ ...patientInfo, name: e.target.value })}
                                                    className="input-field"
                                                    placeholder="VD: Nguyễn Văn A"
                                                />
                                            </div>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div>
                                                    <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2">Ngày sinh (DOB)</label>
                                                    <input
                                                        type="date"
                                                        value={patientInfo.dob || ''}
                                                        onChange={e => {
                                                            const dobValue = e.target.value;
                                                            let calculatedAge = '';
                                                            if (dobValue) {
                                                                const birthYear = new Date(dobValue).getFullYear();
                                                                const currentYear = new Date().getFullYear();
                                                                calculatedAge = birthYear > 0 ? (currentYear - birthYear) : '';
                                                            }
                                                            setPatientInfo({ ...patientInfo, dob: dobValue, age: calculatedAge });
                                                        }}
                                                        className="input-field"
                                                        placeholder="dd/mm/yyyy"
                                                    />
                                                </div>
                                                <div className="grid grid-cols-2 gap-4 mt-4">
                                                    <div>
                                                        <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2 font-black">Giới tính (Gender)</label>
                                                        <div className="flex gap-2">
                                                            {['Nam', 'Nữ'].map(g => (
                                                                <button
                                                                    key={g}
                                                                    onClick={() => setPatientInfo({ ...patientInfo, gender: g })}
                                                                    className={`flex-1 py-3 rounded-2xl font-bold transition-all border-2 text-[10px] uppercase tracking-widest ${patientInfo.gender === g ? 'bg-indigo-600 border-indigo-600 text-white shadow-lg shadow-indigo-200' : 'border-slate-100 dark:border-slate-800 text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800'}`}
                                                                >
                                                                    {g}
                                                                </button>
                                                            ))}
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <label className="block text-xs font-black text-slate-400 uppercase tracking-widest mb-2 font-black">Giọng vùng miền (Dialect)</label>
                                                        <div className="flex gap-2">
                                                            {['North', 'Central', 'South'].map(d => (
                                                                <button
                                                                    key={d}
                                                                    onClick={() => setPatientInfo({ ...patientInfo, dialect: d })}
                                                                    className={`flex-1 py-3 rounded-2xl font-bold transition-all border-2 text-[10px] uppercase tracking-widest ${patientInfo.dialect === d ? 'bg-indigo-600 border-indigo-600 text-white shadow-lg shadow-indigo-200' : 'border-slate-100 dark:border-slate-800 text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800'}`}
                                                                >
                                                                    {d === 'North' ? 'Bắc' : d === 'Central' ? 'Trung' : 'Nam'}
                                                                </button>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex flex-col items-center gap-6 mt-8">
                                        <div className="flex flex-wrap justify-center gap-4 w-full md:w-auto">
                                            <button
                                                onClick={startRecording}
                                                className="btn-primary space-x-2 px-8"
                                            >
                                                <Mic size={20} />
                                                <span>BẮT ĐẦU GHI ÂM</span>
                                                <ArrowRight size={20} />
                                            </button>
                                        </div>

                                        {(!patientInfo.name.trim() || !patientInfo.dob) && (
                                            <p className="text-xs text-slate-400 italic">Thông tin không bắt buộc, nhưng giúp kết quả chính xác hơn.</p>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* STATE: RECORDING */}
                            {appState === STATES.RECORDING && (
                                <div className="text-center space-y-12">
                                    <div className="relative">
                                        <div className={`w-24 h-24 mx-auto rounded-full flex items-center justify-center mb-6 transition-colors ${displayValidTime >= 5.0 ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>
                                            {displayValidTime >= 5.0 ? <CheckCircle size={48} /> : <AlertTriangle size={48} />}
                                        </div>
                                        <div className={`w-64 h-64 mx-auto rounded-full flex items-center justify-center relative overflow-hidden ring-4 transition-all duration-300 ${vadStatus === 'SPEAKING' ? 'bg-green-50 ring-green-200 dark:bg-emerald-900/20 dark:ring-emerald-800/50' : 'bg-slate-100 ring-slate-200 dark:bg-slate-800 dark:ring-slate-700'}`}>
                                            <div className="flex gap-1 items-end h-32 opacity-80 z-0">
                                                {audioVisual.map((h, i) => (
                                                    <motion.div
                                                        key={i}
                                                        animate={{ height: Math.max(10, h * 100) + '%' }}
                                                        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                                                        className={`w-3 rounded-full ${vadStatus === 'SPEAKING' ? 'bg-green-500' : 'bg-slate-400 dark:bg-slate-500'}`}
                                                    />
                                                ))}
                                            </div>
                                            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10 bg-white/40 dark:bg-slate-900/40 rounded-full inset-6">
                                                <div className="text-4xl font-black text-[var(--text-main)] tabular-nums mb-2">
                                                    00:{recordingTime.toString().padStart(2, '0')}
                                                </div>
                                                <div className={`text-[10px] font-bold uppercase tracking-widest ${vadStatus === 'SPEAKING' ? 'text-green-600 dark:text-emerald-400' : 'text-slate-500 dark:text-slate-400'}`}>
                                                    {vadStatus === 'SPEAKING' ? 'Đang phát hiện...' : 'Đang lắng nghe...'}
                                                </div>
                                                {realtimeSai !== null && (
                                                    <motion.div
                                                        initial={{ opacity: 0, y: 10 }}
                                                        animate={{ opacity: 1, y: 0 }}
                                                        className="mt-4 px-3 py-1 bg-indigo-500 text-white text-[10px] font-black rounded-full shadow-lg"
                                                    >
                                                        EST. SAI: {realtimeSai.toFixed(1)}
                                                    </motion.div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        <p className="text-sm font-bold text-slate-500">
                                            Thời lượng giọng nói hợp lệ: <span className={`px-2 py-0.5 rounded-lg ${displayValidTime >= 5.0 ? 'bg-green-100 text-green-600' : 'bg-amber-100 text-amber-600'}`}>{displayValidTime.toFixed(1)}s</span> / 5.0s
                                        </p>

                                        {/* REAL-TIME BIOMARKERS DASHBOARD */}
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto pt-4">
                                            <div className="medical-card p-4 text-left border-sky-100 bg-sky-50/30">
                                                <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1">Live SAI</p>
                                                <p className="text-xl font-black text-sky-600 tabular-nums">{realtimeSai !== null ? realtimeSai.toFixed(1) : '---'}</p>
                                            </div>
                                            <div className="medical-card p-4 text-left">
                                                <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1">Pitch</p>
                                                <p className="text-lg font-bold text-slate-600 tabular-nums">{(realtimeMetrics.mean_f0 || 0).toFixed(0)} <span className="text-[10px]">Hz</span></p>
                                            </div>
                                            <div className="medical-card p-4 text-left">
                                                <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1">Jitter</p>
                                                <p className="text-lg font-bold text-slate-600 tabular-nums">{(realtimeMetrics.jitter_local || 0).toFixed(4)}</p>
                                            </div>
                                            <div className="medical-card p-4 text-left">
                                                <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1">HNR</p>
                                                <p className="text-lg font-bold text-slate-600 tabular-nums">{(realtimeMetrics.hnr || 0).toFixed(1)} <span className="text-[10px]">dB</span></p>
                                            </div>
                                        </div>

                                        <p className="text-[10px] text-slate-400 italic"> (Hệ thống tự động dừng sau {MAX_RECORDING_TIME} giây) </p>
                                    </div>

                                    <button
                                        onClick={stopRecording}
                                        className="bg-red-500 hover:bg-red-600 text-white w-20 h-20 rounded-full shadow-xl hover:scale-110 transition-all flex items-center justify-center mx-auto"
                                    >
                                        <Square size={32} fill="currentColor" />
                                    </button>
                                </div>
                            )}

                            {/* STATE: VALIDATING */}
                            {appState === STATES.VALIDATING && (
                                <div className="text-center space-y-8 py-12">
                                    <div className="relative w-32 h-32 mx-auto">
                                        <div className="absolute inset-0 border-4 border-slate-100 rounded-full"></div>
                                        <div className="absolute inset-0 border-4 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <Activity className="text-sky-500 animate-pulse" size={32} />
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <h3 className="text-2xl font-black text-[var(--text-main)]">Đang kiểm tra chất lượng...</h3>
                                        <p className="text-slate-400">AI đang đánh giá độ ồn, tín hiệu và trường độ</p>
                                    </div>
                                </div>
                            )}

                            {/* STATE: REVIEW (VALID/INVALID) */}
                            {(appState === STATES.REVIEW_VALID || appState === STATES.REVIEW_INVALID) && (
                                <div className="text-center space-y-8">
                                    {appState === STATES.REVIEW_VALID ? (
                                        <div className="w-24 h-24 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg shadow-green-100">
                                            <CheckCircle size={48} />
                                        </div>
                                    ) : (
                                        <div className="w-24 h-24 bg-red-100 text-red-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg shadow-red-100">
                                            <AlertTriangle size={48} />
                                        </div>
                                    )}

                                    <div>
                                        <h2 className={`text-2xl font-black mb-2 ${appState === STATES.REVIEW_VALID ? 'text-[var(--text-main)]' : 'text-red-600'}`}>
                                            {appState === STATES.REVIEW_VALID ? 'Chất lượng tốt!' : 'Cần thu âm lại'}
                                        </h2>
                                        {appState === STATES.REVIEW_INVALID && error && (
                                            <p className="text-red-500 bg-red-50 px-4 py-2 rounded-lg inline-block font-bold text-sm border border-red-100">{error}</p>
                                        )}
                                    </div>

                                    {audioBlob && (
                                        <div className="flex justify-center py-2">
                                            <audio controls src={URL.createObjectURL(audioBlob)} className="h-12 w-full max-w-sm rounded-full bg-slate-100 px-4" />
                                        </div>
                                    )}

                                    {validationData && (
                                        <div className="bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800 p-6 rounded-2xl max-w-sm mx-auto text-left space-y-3">
                                            <h4 className="text-xs font-black uppercase text-slate-400 tracking-widest mb-2">Thông số kỹ thuật</h4>
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-500">Tín hiệu / Nhiễu (SNR)</span>
                                                <span className={`font-bold ${validationData.snr > 15 ? 'text-green-600' : 'text-red-600'}`}>
                                                    {(validationData.snr || 0).toFixed(1)} dB
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-500">Độ rõ (VAD Score)</span>
                                                <span className={`font-bold ${validationData.vad_ratio > 0.3 ? 'text-green-600' : 'text-red-600'}`}>
                                                    {((validationData.vad_ratio || 0) * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-500">Clipping (Vỡ tiếng)</span>
                                                <span className={`font-bold ${validationData.clipping_ratio < 0.01 ? 'text-green-600' : 'text-red-600'}`}>
                                                    {((validationData.clipping_ratio || 0) * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </div>
                                    )}

                                    <div className="flex gap-4 justify-center">
                                        <button
                                            onClick={startRecording}
                                            className="btn-outline flex items-center gap-2"
                                        >
                                            <Mic size={18} /> Thu lại
                                        </button>
                                        {appState === STATES.REVIEW_VALID && (
                                            <button
                                                onClick={handleAnalyze}
                                                className="btn-primary flex items-center gap-2"
                                            >
                                                PHÂN TÍCH NGAY <ArrowRight size={18} />
                                            </button>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* STATE: ANALYZING */}
                            {appState === STATES.ANALYZING && (
                                <div className="text-center space-y-8 py-12">
                                    <div className="relative w-32 h-32 mx-auto">
                                        <div className="absolute inset-0 border-4 border-slate-100 rounded-full"></div>
                                        <div className="absolute inset-0 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <BrainCircuit className="text-indigo-500 animate-bounce" size={32} />
                                        </div>
                                    </div>
                                    <div className="space-y-4">
                                        <h3 className="text-2xl font-black text-indigo-600 animate-pulse">Đang phân tích AI...</h3>
                                        <div className="max-w-xs mx-auto text-left space-y-2">
                                            <div className="flex items-center gap-3 text-slate-500 text-sm">
                                                <div className="w-2 h-2 bg-indigo-500 rounded-full animate-ping" />
                                                Trích xuất 54 đặc trưng quang phổ
                                            </div>
                                            <div className="flex items-center gap-3 text-slate-500 text-sm">
                                                <div className="w-2 h-2 bg-indigo-500 rounded-full animate-ping delay-100" />
                                                Chạy mô hình Random Forest
                                            </div>
                                            <div className="flex items-center gap-3 text-slate-500 text-sm">
                                                <div className="w-2 h-2 bg-indigo-500 rounded-full animate-ping delay-200" />
                                                Tổng hợp báo cáo y khoa
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* STATE: ERROR */}
                            {appState === STATES.ERROR && (
                                <div className="text-center space-y-6 py-12">
                                    <div className="w-24 h-24 bg-red-100 text-red-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg shadow-red-100">
                                        <Server size={48} />
                                    </div>
                                    <h3 className="text-2xl font-black text-red-600">Lỗi hệ thống</h3>
                                    <p className="text-slate-500 max-w-sm mx-auto">{error || "Đã có lỗi xảy ra trong quá trình xử lý."}</p>
                                    <button onClick={() => setAppState(STATES.INFO)} className="btn-outline mx-auto flex items-center gap-2">
                                        <ArrowRight size={18} /> Thử lại
                                    </button>
                                </div>
                            )}

                            {/* RESULTS DASHBOARD */}
                            {appState === STATES.RESULT && analysisData && (
                                <motion.div
                                    key="result"
                                    initial={{ opacity: 0, scale: 0.98 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="space-y-5 max-w-5xl mx-auto w-full"
                                >
                                    <PatientInfoCard info={analysisData.patient_info} metadata={analysisData.metadata} />

                                    {/* HORIZONTAL DASHBOARD HEADER */}
                                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                        {/* SAI SCORE GAUGE CARD */}
                                        <div className="lg:col-span-1 medical-card relative overflow-hidden flex flex-col items-center justify-center py-10 shadow-sky-500/5">
                                            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-sky-500 via-indigo-500 to-emerald-500"></div>
                                            <div className="relative">
                                                <svg className="w-48 h-48 transform -rotate-90">
                                                    <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-slate-100 dark:text-slate-800" />
                                                    <motion.circle
                                                        cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent"
                                                        strokeDasharray={552.92}
                                                        initial={{ strokeDashoffset: 552.92 }}
                                                        animate={{ strokeDashoffset: 552.92 - (552.92 * (analysisData.sai_score || 0)) / 100 }}
                                                        className={analysisData.sai_score > 60 ? "text-rose-500" : analysisData.sai_score > 30 ? "text-amber-500" : "text-emerald-500"}
                                                        strokeLinecap="round"
                                                    />
                                                </svg>
                                                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                                                    <span className="text-4xl font-black text-[var(--text-main)]">{analysisData.sai_score || 0}</span>
                                                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Chỉ số SAI</span>
                                                </div>
                                            </div>
                                            <div className={`mt-6 px-6 py-2 rounded-2xl font-black text-xs border ${analysisData.sai_score > 60 ? "bg-rose-500/10 text-rose-500 border-rose-500/20" :
                                                analysisData.sai_score > 30 ? "bg-amber-500/10 text-amber-500 border-amber-500/20" :
                                                    "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
                                                }`}>
                                                {analysisData.status_msg || (analysisData.sai_score > 60 ? "RỦI RO CAO" : analysisData.sai_score > 30 ? "SAI LỆCH TRUNG BÌNH" : "BÌNH THƯỜNG")}
                                            </div>
                                        </div>

                                        {/* AI EXPLANATION & SPECTRUM */}
                                        <div className="lg:col-span-2 medical-card p-8 flex flex-col justify-between shadow-sky-500/5">
                                            <div className="space-y-4">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-500">
                                                        <BrainCircuit size={20} />
                                                    </div>
                                                    <h2 className="text-sm font-black text-[var(--text-main)] uppercase tracking-[0.2em]">Phân tích từ mô hình AI</h2>
                                                </div>
                                                <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed font-medium italic">
                                                    "{analysisData.explanation}"
                                                </p>

                                                {/* DEVIATED SYSTEMS & RISKS SUMMARY */}
                                                <div className="pt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                                                    <div className="p-4 bg-rose-500/5 border border-rose-500/10 rounded-2xl">
                                                        <h4 className="text-[10px] font-black text-rose-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                                            <AlertTriangle size={12} /> Hệ thống ảnh hưởng
                                                        </h4>
                                                        <div className="flex flex-wrap gap-2">
                                                            {(analysisData.deviated_systems?.length > 0) ? (
                                                                analysisData.deviated_systems.map((s, i) => (
                                                                    <span key={i} className="px-2 py-1 bg-rose-500/10 text-rose-500 text-[10px] font-bold rounded-lg">{s}</span>
                                                                ))
                                                            ) : (
                                                                <span className="text-[10px] font-bold text-slate-400">Chưa ghi nhận sai lệch nghiêm trọng</span>
                                                            )}
                                                        </div>
                                                    </div>
                                                    <div className="p-4 bg-amber-500/5 border border-amber-500/10 rounded-2xl">
                                                        <h4 className="text-[10px] font-black text-amber-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                                            <Shield size={12} /> Rủi ro tiềm ẩn
                                                        </h4>
                                                        <div className="flex flex-wrap gap-2">
                                                            {(analysisData.possible_risks?.length > 0) ? (
                                                                analysisData.possible_risks.map((r, i) => (
                                                                    <span key={i} className="px-2 py-1 bg-amber-500/10 text-amber-500 text-[10px] font-bold rounded-lg">{r}</span>
                                                                ))
                                                            ) : (
                                                                <span className="text-[10px] font-bold text-slate-400">Tầm soát ổn định</span>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                            <SpectrumChart data={analysisData.spectral_data} />
                                        </div>
                                    </div>

                                    {/* COMPREHENSIVE BIOMARKER DASHBOARD (54+ INDICATORS) */}
                                    <div className="medical-card p-8 shadow-sky-500/5 bg-slate-50/30">
                                        <div className="flex items-center justify-between mb-8">
                                            <div className="flex items-center gap-3">
                                                <Activity className="text-sky-500" size={20} />
                                                <h2 className="text-sm font-black text-[var(--text-main)] uppercase tracking-[0.2em]">Hệ thống 54 Acoustic Biomarkers</h2>
                                            </div>
                                            <div className="px-3 py-1 bg-sky-500/10 rounded-full text-[10px] font-black text-sky-600 uppercase">
                                                Scientific Analysis V2.1
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                                            {/* GROUP 1: PITCH & FREQUENCY */}
                                            <div className="space-y-4">
                                                <h3 className="text-[10px] font-black text-indigo-500 uppercase tracking-widest border-l-4 border-indigo-500 pl-3">I. Dây thanh & Tần số</h3>
                                                <div className="space-y-4 pr-4">
                                                    <ProgressBar value={analysisData.details.mean_f0?.value || 0} label="Tần số cơ bản (Mean F0)" min={85} max={255} unit="Hz" />
                                                    <ProgressBar value={analysisData.details.jitter_local?.value || 0} label="Độ rung (Jitter Local)" min={0} max={0.02} unit="%" />
                                                    <ProgressBar value={analysisData.details.f0_std?.value || 0} label="Độ biến thiên F0 (Std)" min={0} max={25} unit="Hz" />
                                                    <ProgressBar value={analysisData.details.pitch_stability?.value || 0} label="Độ ổn định cao độ" min={0.5} max={1.0} unit="idx" />
                                                </div>
                                            </div>

                                            {/* GROUP 2: AMPLITUDE & ENERGY */}
                                            <div className="space-y-4">
                                                <h3 className="text-[10px] font-black text-sky-500 uppercase tracking-widest border-l-4 border-sky-500 pl-3">II. Biên độ & Năng lượng</h3>
                                                <div className="space-y-4 pr-4">
                                                    <ProgressBar value={analysisData.details.rms_energy?.value * 100 || 0} label="Âm lượng (RMS Energy)" min={0} max={100} unit="%" />
                                                    <ProgressBar value={analysisData.details.shimmer_local?.value || 0} label="Độ nhiễu (Shimmer Local)" min={0} max={0.15} unit="dB" />
                                                    <ProgressBar value={analysisData.details.amplitude_mod_index?.value || 0} label="Chỉ số điều chế biên độ" min={0} max={0.4} unit="idx" />
                                                    <ProgressBar value={analysisData.details.mean_amplitude?.value || 0} label="Biên độ trung bình" min={0} max={1.0} unit="idx" />
                                                </div>
                                            </div>

                                            {/* GROUP 3: HARMONIC STRUCTURE */}
                                            <div className="space-y-4">
                                                <h3 className="text-[10px] font-black text-emerald-500 uppercase tracking-widest border-l-4 border-emerald-500 pl-3">III. Cấu trúc hài âm</h3>
                                                <div className="space-y-4 pr-4">
                                                    <ProgressBar value={analysisData.details.hnr?.value || 0} label="Tỷ lệ Hài nhiễu (HNR)" min={0} max={35} unit="dB" />
                                                    <ProgressBar value={analysisData.details.cpp?.value || 0} label="Độ rõ nét hài âm (CPP)" min={0} max={25} unit="dB" />
                                                    <ProgressBar value={analysisData.details.harmonic_richness_factor?.value || 0} label="Độ phong phú hài âm" min={0} max={1.0} unit="idx" />
                                                    <ProgressBar value={Math.abs(analysisData.details.harmonic_spectral_tilt?.value || 0)} label="Độ nghiêng phổ (Tilt)" min={0} max={30} unit="dB" />
                                                </div>
                                            </div>

                                            {/* GROUP 4: SPECTRAL SHAPE */}
                                            <div className="space-y-4">
                                                <h3 className="text-[10px] font-black text-amber-500 uppercase tracking-widest border-l-4 border-amber-500 pl-3">IV. Hình dạng phổ</h3>
                                                <div className="space-y-4 pr-4">
                                                    <ProgressBar value={analysisData.details.spectral_centroid?.value || 0} label="Trọng tâm phổ" min={500} max={4500} unit="Hz" />
                                                    <ProgressBar value={analysisData.details.spectral_rolloff?.value || 0} label="Điểm cuộn phổ (Rolloff)" min={1000} max={6000} unit="Hz" />
                                                    <ProgressBar value={analysisData.details.spectral_flux?.value || 0} label="Biến thiên phổ (Flux)" min={0} max={2.0} unit="idx" />
                                                    <ProgressBar value={Math.abs(analysisData.details.mfcc_1?.value || 0)} label="Đặc trưng MFCC-1" min={0} max={500} unit="val" />
                                                </div>
                                            </div>

                                            {/* GROUP 5: TEMPORAL DYNAMICS */}
                                            <div className="space-y-4">
                                                <h3 className="text-[10px] font-black text-rose-500 uppercase tracking-widest border-l-4 border-rose-500 pl-3">V. Động lực thời gian</h3>
                                                <div className="space-y-4 pr-4">
                                                    <ProgressBar value={analysisData.details.speech_rate?.value || 0} label="Tốc độ phát âm" min={0} max={10} unit="syl/s" />
                                                    <ProgressBar value={analysisData.details.vot?.value || 0} label="Khởi phát thanh (VOT)" min={0} max={0.15} unit="s" />
                                                    <ProgressBar value={analysisData.details.pause_ratio?.value || 0} label="Tỷ lệ ngắt nghỉ" min={0} max={0.5} unit="%" />
                                                    <ProgressBar value={analysisData.details.zcr?.value || 0} label="Tỷ lệ vượt mức 0 (ZCR)" min={0} max={0.2} unit="idx" />
                                                </div>
                                            </div>

                                            {/* GROUP 6: QUALITY INDICES */}
                                            <div className="space-y-4">
                                                <h3 className="text-[10px] font-black text-violet-500 uppercase tracking-widest border-l-4 border-violet-500 pl-3">VI. Chỉ số chất lượng y khoa</h3>
                                                <div className="space-y-4 pr-4">
                                                    <ProgressBar value={analysisData.details.formant_f1?.value || 0} label="F1 Formant" min={300} max={1000} unit="Hz" />
                                                    <ProgressBar value={analysisData.details.breathiness_index?.value || 0} label="Chỉ số tiếng phào" min={0} max={10} unit="val" />
                                                    <ProgressBar value={analysisData.details.hoarseness_index?.value || 0} label="Chỉ số khàn tiếng" min={0} max={10} unit="val" />
                                                    <ProgressBar value={analysisData.details.roughness_index?.value || 0} label="Chỉ số thô ráp" min={0} max={10} unit="val" />
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* RECOMMENDATIONS & FOOTER */}
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        <div className="medical-card p-8 border-rose-500/10 shadow-rose-500/5">
                                            <h3 className="text-[10px] font-black uppercase text-rose-500 tracking-widest mb-6 flex items-center gap-2">
                                                <AlertTriangle size={14} /> Khuyến nghị hướng dẫn y khoa
                                            </h3>
                                            <div className="space-y-3">
                                                {(analysisData.advice || []).map((adv, i) => (
                                                    <div key={i} className="flex gap-4 p-4 rounded-2xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800 transition-all hover:border-rose-500/20">
                                                        <div className="w-6 h-6 rounded-full bg-rose-500/10 text-rose-500 flex items-center justify-center shrink-0 mt-0.5">
                                                            <CheckCircle size={14} />
                                                        </div>
                                                        <p className="text-xs font-bold text-slate-600 dark:text-slate-400 leading-relaxed">{adv}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        <div className="medical-card p-8 bg-sky-600 shadow-2xl shadow-sky-500/20 text-white border-0 flex flex-col justify-between">
                                            <div>
                                                <div className="flex items-center gap-3 mb-6">
                                                    <ShieldCheck size={24} />
                                                    <h3 className="text-xl font-black">Xác nhận chuyên khoa</h3>
                                                </div>
                                                <p className="text-white/80 text-sm font-medium leading-relaxed mb-8">
                                                    Hệ thống AI này được thiết kế để hỗ trợ sàng lọc sớm qua công nghệ phân tích giọng nói. Nếu bạn thấy có dấu hiệu bất thường, hãy liên hệ chuyên gia ngay.
                                                </p>
                                            </div>
                                            <div className="flex gap-4">
                                                <button className="flex-1 py-4 bg-white text-sky-600 rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-sky-50 transition-all shadow-lg">Liên hệ bác sĩ</button>
                                                <button onClick={downloadZip} className="flex-1 py-4 bg-sky-500/50 text-white border border-white/20 rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-sky-500/70 transition-all">Báo cáo (ZIP)</button>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="text-center py-10 opacity-60">
                                        <p className="text-[10px] font-black uppercase tracking-[0.3em] text-[var(--text-muted)] italic">
                                            {analysisData.ai_notice}
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => setAppState(STATES.INFO)}
                                        className="w-full py-6 rounded-3xl bg-slate-100 dark:bg-slate-800 text-[var(--text-main)] font-black text-xs uppercase tracking-[0.3em] hover:bg-slate-200 dark:hover:bg-slate-700 transition-all border border-[var(--card-border)]"
                                    >
                                        Thực hiện kiểm tra mới
                                    </button>
                                </motion.div>
                            )}

                        </motion.div>
                    )}
                </AnimatePresence>
            </main>

            {/* Footer Info - Static at bottom of content primarily */}
            <footer className="w-full py-10 text-center px-8 mt-auto border-t border-[var(--card-border)]/50 bg-[var(--bg-clinical)]">
                <div className="max-w-[1600px] mx-auto flex flex-col md:flex-row justify-between items-center gap-6 opacity-70 hover:opacity-100 transition-opacity duration-500">
                    <p className="text-[10px] font-black uppercase tracking-[0.3em] text-[var(--text-muted)]">
                        © 2026 AI SPEECH ANALYTICS • Ver 4.0-VN
                    </p>
                    <div className="flex flex-wrap justify-center items-center gap-x-8 gap-y-2 text-[10px] font-black uppercase tracking-[0.15em] text-[var(--text-muted)]">
                        <div className="flex items-center gap-2"><Phone size={12} className="text-sky-500" /> 0795277277</div>
                        <div className="flex items-center gap-2"><Mail size={12} className="text-sky-500" /> nguyenduyquangdvfb@gmail.com</div>
                        <div className="flex items-center gap-2">
                            <span className="w-1 h-1 bg-sky-500 rounded-full"></span>
                            Được phát triển bởi Nguyễn Duy Quang
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
}

export default App;
