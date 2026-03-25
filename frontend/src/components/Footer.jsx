// src/components/Footer.js
import React from 'react';
import '../styles/components/Footer.css';
function Footer() {
  return (
    <footer className="custom-footer">
      <div className="container-fluid">
        {/* A single row to align all content, stacking on small screens */}
        <div className="row align-items-center text-center text-lg-start">

          {/* Column 1: App Name and Copyright */}
          <div className="col-lg-4 col-md-12 mb-3 mb-lg-0">
            <p className="footer-brand mb-0">OpenKineticsPredictor</p>
          </div>

          {/* Column 2: Funding Information */}
          <div className="col">
            <div className="d-flex justify-content-center justify-content-lg-end align-items-center">
              <p className="funding-text mb-0">
                EU Horizon Europe (#101080997)
                &ensp;·&ensp;
                Swiss SERI (#23.00232)
                &ensp;·&ensp;
                UKRI (#10083717 &amp; #10080153)
                &ensp;·&ensp;
                FNR (PRIDE21/16763386/CANBIO2)
                &ensp;·&ensp;
                Novo Nordisk Foundation (#NNF10CC1016517)
                &ensp;·&ensp;
                Knut and Alice Wallenberg Foundation
                &ensp;·&ensp;
                EU Horizon 2020 (#686070 &amp; #814650)
              </p>
            </div>
          </div>

        </div>
      </div>
    </footer>
  );
}

export default Footer;
