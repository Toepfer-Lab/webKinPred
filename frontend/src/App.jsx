// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import JobSubmissionForm from './components/job-submission/JobSubmissionForm';
import JobStatus from './components/JobStatus';
import About from './components/About';
import ApiDocs from './components/ApiDocs/ApiDocs';
import Contribute from './components/Contribute';
import NotFound from './components/NotFound';
import Header from './components/Header';
import ProteinBackground from './components/ProteinBackground';
import Footer from './components/Footer';
import './styles/App.css';

function App() {
  return (
    <Router>
      <ProteinBackground />
      <div className="app-container">
        <Header />
        <main className="main-content">
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
    </Router>
  );
}

export default App;
