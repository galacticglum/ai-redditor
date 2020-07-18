import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import Button from 'react-bootstrap/Button';
import './GameOverPanel.css';

class GameOverPanel extends Component {
    onQuitButtonClicked = () => {
        // Redirect to home page
        this.props.history.push('/');
    }

    render() {
        return (
            <div className="panel">
                <h4 className="text-center mt-2 font-weight-bold">{this.props.title}</h4>
                {this.props.content}
                <div className="d-flex flex-row mt-4">
                    <Button size="lg" onClick={this.onQuitButtonClicked}
                        className="w-100 mr-3 action-btn action-btn-quit"
                        disabled={this.props.disabled}
                    >
                        quit
                    </Button>
                    <Button size="lg" onClick={this.props.onPlayAgain}
                        className="w-100 action-btn action-btn-restart"
                        disabled={this.props.disabled}
                    >
                        try again
                    </Button>
                </div> 
            </div>
        )
    }
}

GameOverPanel.defaultProps = {
    disabled: false
}

export default withRouter(GameOverPanel);