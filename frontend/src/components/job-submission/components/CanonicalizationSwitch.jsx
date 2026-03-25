import React from 'react';
import PropTypes from 'prop-types';
import { Form, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { InfoCircleFill } from 'react-bootstrap-icons';

const CanonTooltip = (
  <Tooltip id="canonicalization-tooltip" className="exp-tooltip">
    Canonicalize substrate strings before prediction so equivalent
    SMILES representations are normalized. Turn this off to skip
    canonicalization and follow each method&apos;s native preprocessing.
  </Tooltip>
);

export default function CanonicalizationSwitch({ checked, onChange }) {
  return (
    <Form.Group controlId="canonicalizeSubstrates" className="exp-switch-group">
      <div className="d-flex align-items-center gap-2">
        <Form.Check
          type="switch"
          label="Canonicalize substrates"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="exp-switch"
        />

        <OverlayTrigger placement="right" overlay={CanonTooltip} delay={{ show: 150, hide: 0 }} trigger={['hover', 'focus']}>
          <button
            type="button"
            className="exp-info-btn"
            aria-label="What does ‘Canonicalize substrates’ do?"
          >
            <InfoCircleFill size={16} aria-hidden="true" />
          </button>
        </OverlayTrigger>
      </div>
    </Form.Group>
  );
}

CanonicalizationSwitch.propTypes = {
  checked: PropTypes.bool.isRequired,
  onChange: PropTypes.func.isRequired,
};
