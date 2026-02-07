import { useState, useRef } from 'react';
import axios from 'axios';
import { Mic, Square, Activity, AlertTriangle, CheckCircle, Loader2, Info, Server, Phone, Mail, User, FileText, Download } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
    const [step, setStep] = useState('info'); // 'info' -> 'record' -> 'result'
    const [patientInfo, setPatientInfo] = useState({ name: '', dob: '', cccd: '' });

    const [isRecording, setIsRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [showGuide, setShowGuide] = useState(false);

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const handleInfoSubmit = (e) => {
        e.preventDefault();
        if (patientInfo.name && patientInfo.dob && patientInfo.cccd) {
            setStep('record');
        } else {
            setError("Vui lòng điền đầy đủ thông tin.");
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorderRef.current.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                setAudioBlob(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
            setError(null);
            setResult(null);
        } catch (err) {
            setError("Không thể truy cập Micro. Vui lòng cấp quyền.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const analyzeAudio = async () => {
        if (!audioBlob) return;

        setIsAnalyzing(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('name', patientInfo.name);
        formData.append('dob', patientInfo.dob);
        formData.append('cccd', patientInfo.cccd);

        try {
            const response = await axios.post('http://localhost:8000/analyze', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(response.data);
            setStep('result');
        } catch (err) {
            setError("Phân tích thất bại. Kiểm tra lại Server.");
            console.error(err);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const reset = () => {
        setAudioBlob(null);
        setResult(null);
        setError(null);
        setStep('info');
        setPatientInfo({ name: '', dob: '', cccd: '' });
    };

    const downloadReport = () => {
        if (result?.report_url) {
            window.open(`http://localhost:8000${result.report_url}`, '_blank');
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-800">

            {/* Navbar */}
            <nav className="bg-white shadow-sm border-b border-slate-200 px-6 py-4 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <Activity className="text-blue-600" size={28} />
                    <div>
                        <h1 className="text-xl font-bold text-slate-800">AI Stroke Detection</h1>
                        <p className="text-xs text-slate-500">Phân tích Giọng nói Thông minh</p>
                    </div>
                </div>
            </nav>

            <main className="flex-1 max-w-5xl w-full mx-auto p-6 grid grid-cols-1 md:grid-cols-3 gap-6">

                {/* Left Column: Main Action */}
                <div className="md:col-span-2 space-y-6">
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 relative overflow-hidden min-h-[500px]">

                        <button
                            onClick={() => setShowGuide(!showGuide)}
                            className="absolute top-4 right-4 text-slate-400 hover:text-blue-600 transition-colors"
                        >
                            <Info size={20} />
                        </button>

                        <AnimatePresence mode='wait'>
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0 }}
                                    className="bg-red-50 text-red-600 p-3 rounded-lg mb-6 text-sm flex items-center gap-2 justify-center absolute top-12 left-8 right-12 z-10"
                                >
                                    <AlertTriangle size={18} />
                                    {error}
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* STEP 1: PATIENT INFO FORM */}
                        {step === 'info' && (
                            <motion.div
                                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                                className="flex flex-col h-full justify-center"
                            >
                                <h2 className="text-2xl font-semibold mb-6 text-center">Thông tin Người kiểm tra</h2>
                                <form onSubmit={handleInfoSubmit} className="space-y-4 max-w-md mx-auto w-full">
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Họ và Tên</label>
                                        <input
                                            type="text" required
                                            value={patientInfo.name}
                                            onChange={(e) => setPatientInfo({ ...patientInfo, name: e.target.value })}
                                            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                            placeholder="Nguyễn Văn A"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Ngày sinh</label>
                                        <input
                                            type="date" required
                                            value={patientInfo.dob}
                                            onChange={(e) => setPatientInfo({ ...patientInfo, dob: e.target.value })}
                                            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-700 mb-1">Số CCCD / CMND</label>
                                        <input
                                            type="text" required
                                            value={patientInfo.cccd}
                                            onChange={(e) => setPatientInfo({ ...patientInfo, cccd: e.target.value })}
                                            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                                            placeholder="0791..."
                                        />
                                    </div>
                                    <button
                                        type="submit"
                                        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-medium transition-colors mt-4 flex justify-center items-center gap-2"
                                    >
                                        Tiếp tục <Activity size={18} />
                                    </button>
                                </form>
                            </motion.div>
                        )}

                        {/* STEP 2: RECORDING */}
                        {(step === 'record' || step === 'result') && (
                            <motion.div
                                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                                className="text-center"
                            >
                                <h2 className="text-2xl font-semibold mb-2">Ghi âm Giọng nói</h2>
                                <p className="text-slate-500 mb-8">Người dùng: <span className="font-bold">{patientInfo.name}</span></p>

                                <div className="flex flex-col items-center gap-8 mb-4">
                                    <div className={`relative flex items-center justify-center w-40 h-40 rounded-full transition-all duration-300 ${isRecording ? 'bg-red-50' : 'bg-slate-50'}`}>
                                        {isRecording && (
                                            <motion.span
                                                animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                                                transition={{ repeat: Infinity, duration: 1.5 }}
                                                className="absolute w-full h-full rounded-full border-4 border-red-200"
                                            />
                                        )}
                                        <div className={`w-32 h-32 rounded-full flex items-center justify-center shadow-inner transition-colors ${isRecording ? 'bg-red-100 text-red-600' : 'bg-blue-50 text-blue-600'}`}>
                                            {isRecording ? <Activity size={50} className="animate-pulse" /> : <Mic size={50} />}
                                        </div>
                                    </div>

                                    <div className="flex gap-3">
                                        {!isRecording ? (
                                            !audioBlob ? (
                                                <button
                                                    onClick={startRecording}
                                                    className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-full font-medium transition-colors shadow-lg shadow-blue-200 flex items-center gap-2 transform hover:scale-105"
                                                >
                                                    Bắt đầu Ghi âm
                                                </button>
                                            ) : (
                                                <div className="flex gap-3">
                                                    <button
                                                        onClick={() => { setAudioBlob(null); setResult(null); }}
                                                        className="bg-slate-100 hover:bg-slate-200 text-slate-600 px-6 py-3 rounded-full font-medium transition-colors border border-slate-200"
                                                    >
                                                        Thu lại
                                                    </button>
                                                    <button
                                                        onClick={analyzeAudio}
                                                        disabled={isAnalyzing}
                                                        className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-full font-medium transition-colors shadow-lg shadow-green-200 flex items-center gap-2 disabled:opacity-50 transform hover:scale-105"
                                                    >
                                                        {isAnalyzing ? <Loader2 className="animate-spin" size={20} /> : "Phân tích Ngay"}
                                                    </button>
                                                </div>
                                            )
                                        ) : (
                                            <button
                                                onClick={stopRecording}
                                                className="bg-red-500 hover:bg-red-600 text-white px-8 py-3 rounded-full font-medium transition-colors shadow-lg shadow-red-200 flex items-center gap-2 transform hover:scale-105"
                                            >
                                                <Square size={18} fill="currentColor" /> Dừng Ghi âm
                                            </button>
                                        )}
                                    </div>
                                </div>

                                {/* Results Section */}
                                <AnimatePresence>
                                    {result && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            className="border-t border-slate-100 pt-8 text-left mt-8"
                                        >
                                            <div className="flex justify-between items-center mb-4">
                                                <h3 className="text-slate-400 text-xs uppercase font-bold tracking-wider">Kết Quả Phân Tích</h3>
                                                <button
                                                    onClick={downloadReport}
                                                    className="text-blue-600 text-sm font-medium flex items-center gap-1 hover:underline"
                                                >
                                                    <Download size={16} /> Tải PDF Báo cáo
                                                </button>
                                            </div>

                                            <div className={`p-6 rounded-xl border-l-4 ${result.risk_assessment === 'High Risk' ? 'bg-orange-50 border-orange-500' : 'bg-green-50 border-green-500'} mb-6 shadow-sm`}>
                                                <div className="flex justify-between items-start mb-2">
                                                    <div>
                                                        <span className={`font-bold text-xl block ${result.risk_assessment === 'High Risk' ? 'text-orange-700' : 'text-green-700'}`}>
                                                            {result.risk_assessment === 'High Risk' ? '⚠️ Cảnh báo: Dấu hiệu Bất thường' : '✅ Bình thường: Giọng nói Ổn định'}
                                                        </span>
                                                        <span className="text-sm text-slate-600 mt-1 block">
                                                            {result.risk_assessment === 'High Risk' ? 'Phát hiện rủi ro rối loạn vận ngôn thần kinh.' : 'Không phát hiện dấu hiệu bệnh lý qua giọng nói.'}
                                                        </span>
                                                    </div>
                                                    {result.risk_assessment !== 'High Risk' && <CheckCircle size={32} className="text-green-500" />}
                                                    {result.risk_assessment === 'High Risk' && <AlertTriangle size={32} className="text-orange-500" />}
                                                </div>
                                            </div>

                                            <div className="flex justify-end mt-4">
                                                <button
                                                    onClick={reset}
                                                    className="bg-slate-800 text-white px-6 py-2 rounded-lg text-sm hover:bg-slate-900 transition-colors"
                                                >
                                                    Kiểm tra người khác
                                                </button>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </motion.div>
                        )}



                    </div>

                    <AnimatePresence>
                        {showGuide && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="bg-blue-50 border border-blue-100 p-6 rounded-xl text-sm text-blue-900"
                            >
                                <h3 className="font-bold text-blue-700 mb-2 flex items-center gap-2"><Info size={16} /> Hướng dẫn sử dụng</h3>
                                <ol className="list-decimal pl-5 space-y-1">
                                    <li>Điền đầy đủ thông tin cá nhân.</li>
                                    <li>Đảm bảo bạn đang ở môi trường yên tĩnh.</li>
                                    <li>Nhấn nút <b>Bắt đầu Ghi âm</b>.</li>
                                    <li>Đọc to một câu nói bất kỳ (khoảng 3-5 giây).</li>
                                    <li>Nhấn <b>Dừng</b> và chọn <b>Phân tích Ngay</b>.</li>
                                    <li>Nhấn <b>Tải PDF Báo cáo</b> để lưu kết quả.</li>
                                </ol>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Right Column: Info & Contact */}
                <div className="space-y-6">

                    {/* Tech Info */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                        <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                            <Server className="text-blue-600" size={20} /> Công Nghệ Cốt Lõi
                        </h3>
                        <ul className="space-y-3 text-sm text-slate-600">
                            <li className="flex items-center gap-2"><CheckCircle size={16} className="text-green-500" /> Python FastAPI Backend</li>
                            <li className="flex items-center gap-2"><CheckCircle size={16} className="text-green-500" /> Librosa Audio Processing</li>
                            <li className="flex items-center gap-2"><CheckCircle size={16} className="text-green-500" /> MFCC Feature Extraction</li>
                            <li className="flex items-center gap-2"><CheckCircle size={16} className="text-green-500" /> SVM AI Model (Scikit-learn)</li>
                        </ul>
                    </div>

                    {/* Author / Contact Card */}
                    <div className="bg-gradient-to-br from-slate-900 to-slate-800 p-6 rounded-2xl shadow-lg text-white">
                        <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                            Thông tin Liên hệ
                        </h3>
                        <div className="space-y-3 text-sm text-slate-300">
                            <div className="flex items-center gap-3">
                                <div className="bg-slate-700 p-2 rounded-lg"><User size={16} /></div>
                                <div>
                                    <p className="text-xs text-slate-400">Tác giả</p>
                                    <p className="font-medium text-white">Nguyễn Duy Quang</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="bg-slate-700 p-2 rounded-lg"><Mail size={16} /></div>
                                <div>
                                    <p className="text-xs text-slate-400">Email</p>
                                    <p className="font-medium text-white">nguyenduyquangdvfb@gmail.com</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="bg-slate-700 p-2 rounded-lg"><Phone size={16} /></div>
                                <div>
                                    <p className="text-xs text-slate-400">Điện thoại</p>
                                    <p className="font-medium text-white">0795277227</p>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </main>
        </div>
    )
}

export default App
