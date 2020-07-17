import React, { Component } from 'react';
import Button from 'react-bootstrap/Button';
import './GameOverPanel.css';

export default class GameOverPanel extends Component {
    render() {
        return (
            <div className="panel">
                <h4 className="text-center mt-2 font-weight-bold">{this.props.title}</h4>
                {this.props.content}
                <div className="d-flex flex-row mt-4">
                    <Button className="w-100 mr-3 action-btn action-btn-quit" size="lg">
                        quit
                    </Button>
                    <Button className="w-100 action-btn action-btn-restart" size="lg">
                        play again
                    </Button>
                </div> 
            </div>
        )
    }
}
