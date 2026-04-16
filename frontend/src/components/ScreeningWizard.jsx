import { useState, useRef, useEffect } from 'react';
import { Mic, CheckCircle, ChevronRight, Activity, Play, Square, AlertCircle, Info, Brain, RotateCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const STEPS = [
    {
        id: 'info',
        title: 'Thông tin cá nhân',
        desc: 'Bước 1: Nhập thông tin để theo dõi tiến trình.'
    },
    {
        id: 'vowel',
        title: 'Nguyên âm A',
        desc: 'Bước 2: Kéo dài âm "Aaaa" trong 5 giây.',
        instruction: 'Hãy hít sâu và phát âm "Aaaaa..." đều, thoải mái nhất có thể.'
    },
    {
        id: 'counting',
        title: 'Đếm số',
        desc: 'Bước 3: Đếm từ 1 đến 10.',
        instruction: 'Đếm to, rõ ràng: "Một, Hai, Ba, Bốn, Năm..." với tốc độ vừa phải.'
    },
    {
        id: 'reading',
        title: 'Đọc câu',
        desc: 'Bước 4: Đọc câu mẫu chuẩn hóa.',
        instruction: 'Đọc câu sau: "Mùa thu lá rụng vàng sân, chim hót líu lo trên cành."'
    }
];

export default function ScreeningWizard({ onBack, onComplete }) {
    const [step, setStep] = useState(0);
    const [userData, setUserData] = useState({
        name: '', dob: '', id: '', health: 'normal'
    });
    const [recordings, setRecordings] = useState({
        vowel: null, counting: null, reading: null
    });
    const [isRecording, setIsRecording] = useState(false);
    const [timeLeft, setTimeLeft] = useState(0);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const timerRef = useRef(null);

    // --- RECORDING LOGIC ---
    const startRecording = async (duration = null) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            chunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = e => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
                const currentKey = STEPS[step].id;
                setRecordings(prev => ({ ...prev, [currentKey]: blob }));
                setIsRecording(false);
                stream.getTracks().forEach(t => t.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);

            if (duration) {
                setTimeLeft(duration);
                timerRef.current = setInterval(() => {
                    setTimeLeft(prev => {
                        if (prev <= 1) {
                            stopRecording();
                            return 0;
                        }
                        return prev - 1;
                    });
                }, 1000);
            }

        } catch (e) {
            alert("Lỗi Micro: " + e.message);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
            clearInterval(timerRef.current);
        }
    };

    const handleNext = () => {
        if (step < STEPS.length - 1) {
            setStep(step + 1);
        } else {
            submitAll();
        }
    };

    const submitAll = async () => {
        setIsSubmitting(true);
        const formData = new FormData();
        formData.append('user_id', userData.id);
        formData.append('name', userData.name);
        formData.append('dob', userData.dob);
        formData.append('health_status', userData.health);

        formData.append('task_vowel', recordings.vowel, 'vowel.wav');
        formData.append('task_counting', recordings.counting, 'counting.wav');
        formData.append('task_reading', recordings.reading, 'reading.wav');

        try {
            const API_BASE = (import.meta.env.VITE_API_URL || window.location.origin + '/api').replace(/\/$/, '');
            const res = await axios.post(`${API_BASE}/screening/submit`, formData);
            if (onComplete) onComplete(res.data);
        } catch (e) {
            alert("Lỗi gửi dữ liệu: " + e.message);
            setIsSubmitting(false);
        }
    };

    // --- RENDERERS ---

    const renderInfoStep = () => (
        <div className="space-y-4">
            <h3 className="font-bold text-lg mb-2">Thông tin người khám</h3>
            <input
                placeholder="Mã số / ID (Ví dụ: BN-001)"
                className="w-full p-3 border rounded-xl bg-slate-50"
                value={userData.id}
                onChange={e => setUserData({ ...userData, id: e.target.value })}
            />
            <input
                placeholder="Họ và Tên"
                className="w-full p-3 border rounded-xl bg-slate-50"
                value={userData.name}
                onChange={e => setUserData({ ...userData, name: e.target.value })}
            />
            <input
                type="date"
                className="w-full p-3 border rounded-xl bg-slate-50"
                value={userData.dob}
                onChange={e => setUserData({ ...userData, dob: e.target.value })}
            />

            <div className="pt-2">
                <label className="block text-sm font-medium mb-2 text-slate-600">Trạng thái sức khỏe hôm nay:</label>
                <div className="flex gap-2">
                    {['normal', 'tired', 'stressed'].map(s => (
                        <button
                            key={s}
                            onClick={() => setUserData({ ...userData, health: s })}
                            className={`flex-1 py-2 rounded-lg border text-sm font-bold transition-all
                                ${userData.health === s
                                    ? 'bg-blue-600 text-white border-blue-600 shadow-md transform scale-105'
                                    : 'bg-white text-slate-500 border-slate-200 hover:bg-slate-50'}`}
                        >
                            {s === 'normal' ? '🌞 Bình thường' : s === 'tired' ? '🥱 Mệt mỏi' : '😫 Căng thẳng'}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );

    const renderRecordingStep = (stepInfo) => (
        <div className="text-center py-6">
            <div className="bg-blue-50 p-4 rounded-xl mb-8 inline-block">
                <p className="text-blue-800 text-lg font-medium">"{stepInfo.instruction}"</p>
            </div>

            <div className="flex justify-center mb-6">
                <button
                    onClick={() => isRecording ? stopRecording() : startRecording(stepInfo.id === 'vowel' ? 5 : null)}
                    className={`w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-xl
                        ${isRecording ? 'bg-red-500 animate-pulse' :
                            recordings[stepInfo.id] ? 'bg-green-500' : 'bg-blue-600 hover:scale-105'}`}
                >
                    {isRecording ? <Square size={32} className="text-white" fill="white" /> :
                        recordings[stepInfo.id] ? <CheckCircle size={40} className="text-white" /> :
                            <Mic size={40} className="text-white" />}
                </button>
            </div>

            {isRecording && stepInfo.id === 'vowel' && (
                <div className="text-4xl font-mono font-bold text-slate-300">
                    00:0{timeLeft}
                </div>
            )}

            {recordings[stepInfo.id] && !isRecording && (
                <p className="text-green-600 font-bold animate-bounce">Đã thu âm xong!</p>
            )}
        </div>
    );

    const currentStep = STEPS[step];
    const canContinue = () => {
        if (step === 0) return userData.id && userData.name;
        return !!recordings[currentStep.id];
    };

    if (isSubmitting) {
        return (
            <div className="flex flex-col items-center justify-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                <p className="text-slate-500 font-medium">Đang phân tích tổng hợp...</p>
            </div>
        );
    }

    return (
        <div className="max-w-2xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden border border-slate-100 mt-10">
            {/* Header Progress */}
            <div className="bg-slate-900 p-6 text-white flex justify-between items-center">
                <div>
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Activity className="text-blue-400" /> Chế độ Tầm soát Chuẩn
                    </h2>
                    <p className="text-slate-400 text-sm">Bước {step + 1}/{STEPS.length}: {currentStep.title}</p>
                </div>
                <div className="text-xs font-mono bg-slate-800 px-3 py-1 rounded-full">
                    {Math.round(((step + 1) / STEPS.length) * 100)}%
                </div>
            </div>

            {/* Content */}
            <div className="p-8 min-h-[400px]">
                <AnimatePresence mode='wait'>
                    <motion.div
                        key={step}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                    >
                        {step === 0 ? renderInfoStep() : renderRecordingStep(currentStep)}
                    </motion.div>
                </AnimatePresence>
            </div>

            {/* Footer */}
            <div className="p-6 border-t bg-slate-50 flex justify-between">
                <button
                    onClick={step === 0 ? onBack : () => setStep(step - 1)}
                    className="px-6 py-2 text-slate-500 font-medium hover:text-slate-800"
                >
                    Quay lại
                </button>
                <button
                    onClick={handleNext}
                    disabled={!canContinue()}
                    className="px-8 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all"
                >
                    {step === STEPS.length - 1 ? 'Hoàn tất & Phân tích' : 'Tiếp tục'} <ChevronRight size={18} />
                </button>
            </div>
        </div>
    );
}
