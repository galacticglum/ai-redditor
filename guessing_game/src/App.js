import React, { Component } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';

import HomePage from './pages/HomePage';
import GamePage from './pages/game/GamePage';
import './App.css';

import axios from 'axios';

// Add response time
axios.interceptors.request.use(x => {
    x.meta = x.meta || {};
    x.meta.startTime = new Date().getTime();
    return x;
});

axios.interceptors.response.use(x => {
    x.duration = new Date().getTime() - x.config.meta.startTime;
    return x;
}, x => {
    x.duration = new Date().getTime() - x.config.meta.startTime;
    throw x;
})

export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

export default class App extends Component {
    render() {
        return (
            <BrowserRouter>
                <Switch>
                    <Route path="/" component={HomePage} exact />
                    <Route path="/play" component={GamePage} exact />
                </Switch>
            </BrowserRouter>
        )
    }
}