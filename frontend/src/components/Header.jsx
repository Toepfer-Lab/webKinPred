// Header.js
import React from 'react';
import { Navbar, Container, Nav } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import '../styles/components/navbar.css';
import { useTheme } from '../context/ThemeContext';

const SunIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="4.5" />
    <line x1="12" y1="2" x2="12" y2="4.5" />
    <line x1="12" y1="19.5" x2="12" y2="22" />
    <line x1="4.22" y1="4.22" x2="5.93" y2="5.93" />
    <line x1="18.07" y1="18.07" x2="19.78" y2="19.78" />
    <line x1="2" y1="12" x2="4.5" y2="12" />
    <line x1="19.5" y1="12" x2="22" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.93" y2="18.07" />
    <line x1="18.07" y1="5.93" x2="19.78" y2="4.22" />
  </svg>
);

const MoonIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);

function ThemeToggle({ theme, toggleTheme, extraClass = '' }) {
  return (
    <button
      className={`theme-toggle-btn${extraClass ? ' ' + extraClass : ''}`}
      onClick={toggleTheme}
      aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      <span className="theme-toggle-track">
        <span className="theme-toggle-thumb">
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </span>
      </span>
    </button>
  );
}

function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <Navbar expand="lg" className="custom-navbar">
      <Container>
        {/* Brand always on the left */}
        <Navbar.Brand as={Link} to="/">OpenKineticsPredictor</Navbar.Brand>

        {/* Collapse contains nav links (immediately after brand on desktop) */}
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav>
            <Nav.Link as={Link} to="/track-job">Track Job</Nav.Link>
            <Nav.Link as={Link} to="/api-docs">API</Nav.Link>
            <Nav.Link as={Link} to="/contribute">Contribute</Nav.Link>
            <Nav.Link as={Link} to="/about">About</Nav.Link>
          </Nav>
          {/* Desktop-only toggle — far right inside collapse */}
          <ThemeToggle theme={theme} toggleTheme={toggleTheme} extraClass="d-none d-lg-inline-flex ms-auto" />
        </Navbar.Collapse>

        {/* Mobile-only toggle — always visible before hamburger */}
        <ThemeToggle theme={theme} toggleTheme={toggleTheme} extraClass="d-lg-none ms-auto" />
        <Navbar.Toggle className="ms-2" aria-controls="basic-navbar-nav" />
      </Container>
    </Navbar>
  );
}

export default Header;
