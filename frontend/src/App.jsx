// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import JobSubmissionForm from './components/job-submission/JobSubmissionForm';
import JobStatus from './components/JobStatus';
import About from './components/About';
import ApiDocs from './components/ApiDocs/ApiDocs';
import Contribute from './components/Contribute';
import NotFound from './components/NotFound';
import Header from './components/Header';
import GpuStatus from './components/GpuStatus';
import ProteinBackground from './components/ProteinBackground';
import Footer from './components/Footer';
import './styles/App.css';

function AppContent() {
  const location = useLocation();
  const isTrackPage = location.pathname.startsWith('/track-job');
  const showGpuStatus = location.pathname === '/' || location.pathname.startsWith('/track-job');
  const gpuLayout = isTrackPage ? 'track' : 'home';

  return (
    <>
      <ProteinBackground />
      <div className={`app-container${isTrackPage ? ' app-container--track' : ''}`}>
        <Header />
        <main className={`main-content${isTrackPage ? ' main-content--track' : ''}`}>
          {showGpuStatus && <GpuStatus layout={gpuLayout} />}
          <Routes>
            <Route path="/" element={<JobSubmissionForm />} />
            <Route path="/track-job/:public_id" element={<JobStatus />} />
            <Route path="/track-job" element={<JobStatus />} />
            <Route path="/about" element={<About />} />
            <Route path="/api-docs" element={<ApiDocs />} />
            <Route path="/contribute" element={<Contribute />} />
            <Route path="*" element={<NotFound />} />
            {/* <Route path="/evaluation" element={<Evaluation />} /> */}
          </Routes>
        </main>
        <Footer />
      </div>
    </>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
