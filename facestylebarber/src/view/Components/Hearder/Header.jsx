import React from 'react'
import './Header.css';

import logo from './Logo.jpg';

export default function Header() {
  return (
    <div>
      <header>
        <article id='art01'>
          <img src={logo} alt='logo' />
        </article>
        <article id='art02'>
        <div className="divider"></div>
        </article>
        <article id='art03'>
          <span className='letra-grande'>FaceStyle Barber</span>
        </article>
      </header>        
    </div>
  )
}
