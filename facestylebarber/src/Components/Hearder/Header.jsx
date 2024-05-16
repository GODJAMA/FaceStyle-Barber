import React from 'react'
import './Header.css';
import logo from '../../img/Imagen de WhatsApp 2024-04-27 a las 15.05.54_3e31bc63.jpg';

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
          <span className='letra-grande'>FaceSyle Barber</span>
        </article>
        {/* <hr/> */}
      </header>
        <article id='art04'>         
          <nav>
            <ul>
              <li>Barberia</li>
              <li>Mision y Vision</li>
              <li>Estado De Arte</li>
              <li>Memoria Tenica</li>
              <li>Ficha Tecnica</li>
            </ul>
          </nav>
        </article>
    </div>
  )
}
