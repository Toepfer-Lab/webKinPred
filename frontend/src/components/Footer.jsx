// src/components/Footer.js
import React from 'react';
import '../styles/components/Footer.css';

const funders = [
  { name: 'EU Horizon Europe', grant: '#101080997' },
  { name: 'Swiss SERI', grant: '#23.00232' },
  { name: 'UKRI', grant: '#10083717 & #10080153' },
  { name: 'FNR', grant: 'PRIDE21/16763386/CANBIO2' },
  { name: 'Novo Nordisk Foundation', grant: '#NNF10CC1016517' },
  { name: 'Knut & Alice Wallenberg Foundation', grant: null },
  { name: 'EU Horizon 2020', grant: '#686070 & #814650' },
  { name: 'National Key R&D China', grant: '2025YFA0922700' },
];

function Footer() {
  return (
    <footer className="custom-footer">
      <div className="container-fluid">
        <div className="footer-inner">

          {/* Brand */}
          <p className="footer-brand mb-0">OpenKineticsPredictor</p>

          {/* Funding */}
          <div className="funding-section">
            <span className="funding-label">Funded by</span>
            <div className="funding-badges">
              {funders.map(({ name, grant }) => (
                <span key={name} className="funding-badge">
                  <span className="badge-name">{name}</span>
                  {grant && <span className="badge-grant">{grant}</span>}
                </span>
              ))}
            </div>
          </div>

        </div>
      </div>
    </footer>
  );
}

export default Footer;
