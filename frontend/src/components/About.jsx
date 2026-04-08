// src/components/About.js
import { useState } from 'react';
import { Button, Form } from 'react-bootstrap';
import './ApiDocs/ApiDocs.css';

const teamInstitutions = [
  {
    institution: 'Digital Metabolic Twin Center, University of Galway',
    location: 'Galway, Ireland',
    members: ['Ronan Fleming', 'Saleh Alwer'],
  },
  {
    institution: 'Faculty of Science, Technology and Medicine, University of Luxembourg',
    location: 'Esch-sur-Alzette, Luxembourg',
    members: ['Thomas Sauter', 'Hugues Escoffier'],
  },
  {
    institution: 'Systems Biology, Department of Life Sciences, Chalmers University of Technology',
    location: 'Gothenburg, Sweden',
    members: ['Eduard Kerkhoven'],
  },
  {
    institution: 'Luo Laboratory, Center for Synthetic Biochemistry, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences',
    location: 'Shenzhen, China',
    members: ['Han Yu', 'Xiaozhou Luo'],
  },
  {
    institution: 'Maranas Group, Department of Chemical Engineering, The Pennsylvania State University',
    location: 'University Park, PA, USA',
    members: ['Costas D. Maranas', 'Veda Sheersh Boorla', 'Somtirtha Santar'],
  },
  {
    institution: 'Shenzhen Zelixir Biotech Co. Ltd',
    location: 'Shenzhen, Guangdong, China',
    members: ['Liangzhen Zheng'],
  },
  {
    institution: 'School of Physics, Shandong University',
    location: 'Jinan, China',
    members: ['Zechen Wang'],
  },
];

const About = () => {
  const [copied, setCopied] = useState(false);
  const citationText = "OpenKineticsPredictor: ....";

  const copyCitation = () => {
    navigator.clipboard.writeText(citationText)
      .then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
      })
      .catch(err => console.error('Failed to copy: ', err));
  };

  return (
    <div className="api-docs-page">
      <div className="container pt-4 pb-5" style={{ color: '#e8e6f0', maxWidth: '900px' }}>

        {/* Header */}
        <div style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1.5rem', marginBottom: '2.5rem' }}>
          <h2 style={{ fontWeight: 700, color: '#ffffff', marginBottom: 0 }}>About OpenKineticsPredictor</h2>
        </div>

        {/* Team Section */}
        <section style={{ marginBottom: '3rem' }}>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '1rem' }}>
            {teamInstitutions.map((entry, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '10px',
                  padding: '1.1rem 1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.6rem',
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#ffffff', fontSize: '0.95rem', lineHeight: 1.4, marginBottom: '0.2rem' }}>
                    {entry.institution}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'rgba(180,160,255,0.8)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span style={{ opacity: 0.6 }}>&#x1F4CD;</span>
                    {entry.location}
                  </div>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.25rem' }}>
                  {entry.members.map((member, mIdx) => (
                    <span
                      key={mIdx}
                      style={{
                        background: 'rgba(180,160,255,0.12)',
                        border: '1px solid rgba(180,160,255,0.25)',
                        color: '#d4c5ff',
                        borderRadius: '20px',
                        padding: '0.2rem 0.7rem',
                        fontSize: '0.82rem',
                        fontWeight: 500,
                      }}
                    >
                      {member}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Citation Section */}
        <section style={{ marginBottom: '2rem' }}>
          <h4 style={{
            fontWeight: 600,
            color: '#ffffff',
            borderLeft: '3px solid rgba(180,160,255,0.7)',
            paddingLeft: '0.75rem',
            marginBottom: '1rem',
          }}>
            Citation
          </h4>
          <Form.Group controlId="citationText">
            <Form.Control as="textarea" rows={3} readOnly value={citationText} style={{ background: 'rgba(13,11,26,0.8)', color: '#dcd8f0', border: '1px solid rgba(255,255,255,0.12)' }} />
          </Form.Group>
          <Button variant="secondary" className="mt-2" onClick={copyCitation}>
            {copied ? "Copied!" : "Copy Citation"}
          </Button>
        </section>

      </div>
    </div>
  );
};

export default About;
