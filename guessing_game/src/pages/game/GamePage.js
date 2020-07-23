import React, { Component, useState } from 'react';
import Alert from 'react-bootstrap/Alert';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import { CSSTransition } from 'react-transition-group';

import ConfigPanel from './ConfigPanel';
import GameOverPanel from './GameOverPanel';
import './GamePage.css';

import axios from 'axios';
import { API_BASE_URL } from '../../App';

function DismissibleAlert(props) {
    const [show, setShow] = useState(true);

    if (show) {
        return (
            <Alert {...props} onClose={() => setShow(false)} dismissible>
                {props.children}
            </Alert>
        )
    }

    return null;
}

function Record(props) {
    let recordView = null;
    let postPanelClass = "post-panel";
    switch (props.type) {
        case 'tifu':
            recordView = (
                <h4 className="text-center mt-2 font-weight-bold">
                    {props.data.post_title}
                </h4>
            );
            break;
        case 'wp':
            recordView = (
                <h4 className="text-center mt-2 font-weight-bold">
                    {props.data.prompt}
                </h4>
            );
            break;
        case 'phc':
            recordView = (
                <div className="d-flex flex-column">
                    <span className="font-weight-bold phc-username">{props.data.author_username}</span>
                    <p className="phc-comment-text mb-0">{props.data.comment}</p>
                </div>
            );
            postPanelClass += " phc-panel"
            break;
        default:
            return ( 'invalid record type!' )
    }

    return (
        <div className={`${postPanelClass} ${props.className}`}>
            {recordView}
        </div>  
    )
}

function randomFloat(min, max) {
    return Math.random() * (max - min) + min;
}

// The number of milliseconds to show the guess results for.
// In the meantime, the next record is fetched from the server,
// which may take longer than the alloted durartion.
const GUESS_RESULT_DURATION_MS = 500;

export default class GamePage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            ...this.getInitialState(),
            // Round ID is used to uniquely identify the current round
            roundId: 0
        }
    }

    getInitialState = () => {
        return {
            gameConfig: {},
            currentRecord: undefined,
            isLoadingRecord: false,
            hasError: false,
            hasGuessed: false,
            isGuessCorrect: false,
            guessIsGenerated: false,
            guessingTimeCountdownStarted: false,
            guessingTimeCountdownFinished: false,
            score: 0,
            isGameover: false,
            // A manual override to hide the config panel
            hideConfigPanel: false,
        };
    }

    onConfigPanelReady = (configState) => {
        // Map the record types from a (string, bool) key/value pair to
        // an array of strings indicating the types to sample from.
        let recordTypes = [];
        for (const [key, value] of Object.entries(configState.recordTypes)) {
            if (!value) continue;
            recordTypes.push(key);
        }

        this.setState({
            gameConfig: {
                ...configState,
                recordTypes: recordTypes
            }
        }, () => {
            this.nextRecord(0);
        });
    }

    nextRecord = (waitDurationMs=GUESS_RESULT_DURATION_MS, onComplete=null) => {
        if (this.state.hasGuessed && !this.state.isGuessCorrect && !this.state.isGameover) {
            this.onGameOver();
        } else {
            // Randomly select record type
            const recordTypes = this.state.gameConfig.recordTypes;
            const recordType = recordTypes[Math.floor(Math.random() * recordTypes.length)];
            const isRecordGenerated = Math.random() < randomFloat(0.40, 0.60);
            this.fetchRecord(recordType, isRecordGenerated, waitDurationMs, onComplete);
        }
    }

    fetchRecord = (recordType, isGenerated, waitDurationMs=GUESS_RESULT_DURATION_MS, onComplete=null) => {
        const requestData = {
            // A "custom" record refers to a user-generated record
            'is_custom': false,
            // If a record is not generated, it means it was not 'written'
            // by the GPT2 model (i.e. it is written by a human).
            'is_generated': isGenerated,
        };

        // Reset state variables
        this.setState({
            hasError: false,
            isLoadingRecord: true,
            roundId: this.state.roundId + 1
        });

        axios.post(`${API_BASE_URL}/r/${recordType}/random`, {
            ...requestData,    
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            let waitTime = waitDurationMs - response.duration;
            const currentRecord = {
                data: response.data,
                type: recordType
            };

            setTimeout(() => {
                const maxGuessingTimeEnabled = this.state.gameConfig.maxGuessingTimeEnabled;
                this.setState({
                    currentRecord: currentRecord,
                    hasGuessed: false,
                    isLoadingRecord: false,
                    guessingTimeCountdownStarted: maxGuessingTimeEnabled,
                    guessingTimeCountdownFinished: false
                });

                if (onComplete) {
                    onComplete();
                }

                if (this.state.gameConfig.maxGuessingTimeEnabled) {
                    const currentRoundId = this.state.roundId;
                    setTimeout(() => {
                        if (this.state.hasGuessed || this.state.roundId !== currentRoundId) return;

                        setTimeout(() => {
                            this.onGameOver();
                        }, GUESS_RESULT_DURATION_MS);

                        this.setState({
                            guessingTimeCountdownFinished: true
                        });
                    }, this.state.gameConfig.maxGuessingTime * 1000);
                }
            }, Math.max(0, waitTime));
        })
        .catch(error => {
            this.setState({hasError: true, isLoadingRecord: false});
            console.log(error);
        });
    }

    onGuessButtonClicked = (guessIsGenerated) => {
        const isCorrect = guessIsGenerated === this.state.currentRecord.data.is_generated;
        let newScore = this.state.score;
        if (isCorrect) {
            console.log('WOWOWOWOWOOW AMAZING YOU GOT IT RIGHT');
            newScore += 1;
        } else {
            console.log('you fucking suck');
        }

        this.setState({
            hasGuessed: true,
            isGuessCorrect: isCorrect,
            guessIsGenerated: guessIsGenerated,
            guessingTimeCountdownStarted: false,
            guessingTimeCountdownFinished: false,
            score: newScore
        });

        setTimeout(() => {
            this.nextRecord();
        }, GUESS_RESULT_DURATION_MS);
    }

    onGameOver = () => {
        this.setState({
            isGameover: true
        });

        console.log('GAMEOVER!');
    }

    recordButtonText = (guessIsGenerated, defaultText) => {
        return this.state.hasGuessed && this.state.guessIsGenerated === guessIsGenerated ? (
            this.state.isGuessCorrect ? 'correct!' : 'incorrect!'
        ) : defaultText;
    }

    recordButtonResultClassName = (guessIsGenerated) => {
        if (this.state.hasGuessed) {
            return this.state.guessIsGenerated === guessIsGenerated ? 'highlight-border' : 'opacity-30';
        }
        
        if (this.state.guessingTimeCountdownFinished) {
            return 'opacity-30';
        }

        return '';
    }

    playAgain = () => {
        this.setState({
            ...this.getInitialState(),
            gameConfig: this.state.gameConfig,
            hideConfigPanel: true,
            isGameover: true,
            guessingTimeCountdownFinished: this.state.guessingTimeCountdownFinished,
            score: this.state.score
        });

        this.nextRecord(0, () => {
            this.setState({
                hideConfigPanel: false,
                isGameover: false,
                guessingTimeCountdownFinished: false,
                score: 0
            });
        });
    }

    getGameoverStatusText = () => {
        // [a, b] intervals where null indicates +/- infinity.
        const INTERVALS = [
            { min: 0, max: 0, text: 'bruh' },
            { min: 1, max: 4, text: 'you suck' },
            { min: 5, max: 7, text: 'wow! you are just about average!' },
            { min: 8, max: 9, text: 'incredible! your IQ is probably greater than 100!' },
            { min: 10, max: 14, text: 'eureka! you are a GOD!' },
            { min: 15, max: 24, text: 'BLOOBLE!' },
            { min: 25, max: null, text: 'UNSTOPABBLE!!' }
        ];

        const score = this.state.score;
        for (const interval of INTERVALS) {
            if (interval.min < score || score > interval.max) continue;
            return interval.text;
        }

        return 'you suck';
    }

    render() {
        const hasRecord = this.state.currentRecord !== undefined
        const showConfigPanel = !hasRecord && !this.state.hideConfigPanel;
        const isGameover = this.state.isGameover;

        return (
            <Container className="w-100 h-100 d-flex flex-column">
                <div>
                    <CSSTransition
                        in={this.state.guessingTimeCountdownStarted}
                        timeout={1000 * this.state.gameConfig.maxGuessingTime}
                        classNames="timer-progress-bar"
                    >
                        {!showConfigPanel && this.state.gameConfig.maxGuessingTimeEnabled ? (
                            <div key="timer-progress-bar" className="timer-progress-bar" style={{
                                '--timer-progress-bar-duration': `${this.state.gameConfig.maxGuessingTime}s`
                            }} />
                        ) : <div key="css-transition-placeholder" />}
                    </CSSTransition>
                    {!showConfigPanel && this.state.gameConfig.maxGuessingTimeEnabled && (
                        <div className="timer-progress-bar timer-progress-bar-background w-100" />
                    )}
                </div>
                <Row className="mt-auto">
                    <Col sm="12" md="8" lg="6" className="mx-auto">
                        <div id="view-wrapper">
                            {
                                this.state.hasError ? 
                                    (<DismissibleAlert variant="danger" className="error-alert">
                                        <Alert.Heading>server go brrr</Alert.Heading>
                                        This is probably an issue with our servers. Please try again later.
                                    </DismissibleAlert>)
                                : null
                            }
                            {
                                showConfigPanel ? 
                                    (<ConfigPanel action={this.onConfigPanelReady} disabled={this.state.isLoadingRecord} />)
                                : (
                                    !isGameover && hasRecord && (<div>  
                                        <Record type={this.state.currentRecord.type}
                                            data={this.state.currentRecord.data}
                                            className={this.state.hasGuessed ? 
                                                (this.state.isGuessCorrect ? 
                                                    'post-correct' : 'post-incorrect'
                                                ) : ''
                                            }
                                        />
                                        <div className="d-flex flex-row mt-4">
                                            <Button onClick={() => this.onGuessButtonClicked(true)} disabled={this.state.hasGuessed 
                                                || this.state.guessingTimeCountdownFinished} size="lg"
                                                className={`w-100 mr-3 select-btn select-ai-btn 
                                                    ${this.recordButtonResultClassName(true)}`
                                                }
                                            >
                                                {this.recordButtonText(true, 'robot')}
                                            </Button>
                                            <Button onClick={() => this.onGuessButtonClicked(false)} disabled={this.state.hasGuessed
                                                || this.state.guessingTimeCountdownFinished} size="lg"
                                                className={`w-100 select-btn select-human-btn
                                                    ${this.recordButtonResultClassName(false)}`
                                                }
                                            >
                                                {this.recordButtonText(false, 'human')}
                                            </Button>
                                        </div>
                                    </div>)
                                )
                            }
                            {
                                isGameover && (
                                    <GameOverPanel
                                        title={this.state.guessingTimeCountdownFinished ? 'times up!' : this.getGameoverStatusText()}
                                        content={(
                                            <div className="my-5 gameover-results">
                                                <h4 className="mt-3">
                                                    <span className={this.state.score === 0 ? 'text-danger' : 'text-success'}>
                                                        {this.state.score}
                                                    </span> point{this.state.score !== 1 && (<span>s</span>)}
                                                </h4>
                                                <p>{this.state.guessingTimeCountdownFinished ?
                                                    'next time try to think faster' :
                                                    'bested by an artificial intelligence'
                                                }</p>
                                            </div>
                                        )}
                                        onPlayAgain={this.playAgain}
                                        disabled={this.state.isLoadingRecord}
                                    />
                                )
                            }
                        </div>    
                    </Col>
                </Row>

                {
                    (showConfigPanel ? 
                        (<div className="mx-auto mt-auto py-4 text-center warn-footer">
                            <span className="font-weight-bold">warning:</span> posts are not reviewed and are probably <a 
                            href="https://www.pornsfw.com/" className="text-link">NSFW</a>, 
                            use at your own risk!
                        </div>)
                    : null)
                }    
            </Container>
        );
    }
}
