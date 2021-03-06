import React from 'react';
import { NavLink } from 'react-router-dom';
import './App.css';

/**
 * NAME: Header
 * 
 * PURPOSE: Implements a reusable Navigation menu bar react component.
 *
 * SUMMARY: 
 *  - Display header
 *  - Create NavLink components for each route in the web app 
 *      - Home Page(/) 
 *      - Todays Predictions(/todays-games) 
 *      - Predictions by Date(/predict-by-date)
 * 
 * AUTHOR:
 *  - Brandan Quinn
 * 
 * DATE:
 *  10:57am 4/3/19
 */
const Header = () => (
    <header class="Navbar">
        <h1 class="Sitename">QML NBA Predictions</h1>
        <hr /> 
        <NavLink to="/" activeClassName="Link" class="Link" exact={true}>Home Page</NavLink>
        <NavLink to="/todays-games" activeClassName="Link" class="Link">Todays Predictions</NavLink>
        <NavLink to="/get-date" activeClassName="Link" class="Link">Predictions By Date</NavLink>
        <NavLink to="/matchup-prediction" activeClassName="Link" class="Link">Matchup Prediction</NavLink>
        <hr />
    </header>
);

export default Header;