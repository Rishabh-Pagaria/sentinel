/**
 * Sentinel Gmail Add-on - Server-side Apps Script
 * 
 * This script handles Gmail context and extracts email data when a message is opened.
 * It sends the data to the Sentinel backend for phishing detection analysis.
 */

// Configuration
const CONFIG = {
  // For local testing, use ngrok or similar to expose localhost
  // Replace with your actual backend URL
  BACKEND_URL: 'https://triumphal-nonempathically-tabetha.ngrok-free.dev/classify',
};

/**
 * Triggered when a user opens a Gmail message.
 * Returns a card to display in the side panel.
 * 
 * @param {Object} e - Event object containing Gmail message context
 * @return {Card} A card UI to display in the side panel
 */
function onGmailMessageOpen(e) {
  // Get the message ID from the event
  var messageId = e.gmail.messageId;
  var accessToken = e.gmail.accessToken;
  
  // Fetch email details
  var emailData = getEmailData(messageId, accessToken);
  
  // Build the sidebar card
  return buildSidebarCard(emailData);
}

/**
 * Extracts email data using Gmail API
 * 
 * @param {string} messageId - The Gmail message ID
 * @param {string} accessToken - OAuth access token
 * @return {Object} Email data object with to, subject, from, body, and raw content
 */
function getEmailData(messageId, accessToken) {
  try {
    // Use Gmail API to get message details
    var message = GmailApp.getMessageById(messageId);
    
    if (!message) {
      return {
        subject: "Unable to load email",
        from: "Unknown",
        body: "Could not retrieve email content.",
        rawBody: "",
        error: true
      };
    }
    
    // Extract email components
    var subject = message.getSubject() || "(No Subject)";
    var from = message.getFrom() || "Unknown Sender";
    var to = message.getTo() || "Unknown Recipient";
    var plainBody = message.getPlainBody() || "";
    var htmlBody = message.getBody() || "";
    
    // Use plain body if available, otherwise strip HTML from body
    var bodyText = plainBody || stripHtmlTags(htmlBody);
    
    // Limit body length for display (first 1000 chars)
    var displayBody = bodyText.length > 1000 
      ? bodyText.substring(0, 1000) + "..." 
      : bodyText;
    
    return {
      to: to,
      subject: subject,
      from: from,
      body: displayBody,
      rawBody: bodyText, // Full body for API analysis
      error: false
    };
    
  } catch (error) {
    Logger.log("Error fetching email data: " + error.toString());
    return {
      to: "Unknown",
      subject: "Error",
      from: "Unknown",
      body: "Error: " + error.toString(),
      rawBody: "",
      error: true
    };
  }
}

/**
 * Strips HTML tags from a string
 * 
 * @param {string} html - HTML string
 * @return {string} Plain text
 */
function stripHtmlTags(html) {
  return html.replace(/<[^>]*>/g, ' ')
             .replace(/\s+/g, ' ')
             .trim();
}

/**
 * Builds the sidebar card UI
 * 
 * @param {Object} emailData - Email data object
 * @return {Card} Card UI for the sidebar
 */
function buildSidebarCard(emailData) {
  var card = CardService.newCardBuilder();
  
  // Header
  var header = CardService.newCardHeader()
    .setTitle('Sentinel Email Analyzer')
    .setSubtitle('Phishing Detection');
  
  card.setHeader(header);
  
  // Email details section
  var section = CardService.newCardSection();
  
  // Subject widget
  section.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Subject')
      .setContent(emailData.subject)
      .setMultiline(true)
  );

  // To widget
  section.addWidget(
    CardService.newKeyValue()
      .setTopLabel('To')
      .setContent(emailData.to)
      .setMultiline(true)
  );
  
  // From widget
  section.addWidget(
    CardService.newKeyValue()
      .setTopLabel('From')
      .setContent(emailData.from)
      .setMultiline(true)
  );
  
  // Divider
  section.addWidget(CardService.newDivider());
  
  // Body preview widget
  section.addWidget(
    CardService.newTextParagraph()
      .setText('<b>Email Body:</b>')
  );
  
  section.addWidget(
    CardService.newTextParagraph()
      .setText(emailData.body) // Show full display body (already limited to 1000 chars)
  );
  
  // Add analyze button
  if (!emailData.error) {
    section.addWidget(CardService.newDivider());
    
    var analyzeButton = CardService.newTextButton()
      .setText('Analyze for Phishing')
      .setOnClickAction(
        CardService.newAction()
          .setFunctionName('analyzeEmail')
          .setParameters({
            'subject': emailData.subject,
            'from': emailData.from,
            'body': emailData.rawBody
          })
      );
    
    section.addWidget(
      CardService.newButtonSet()
        .addButton(analyzeButton)
    );
  }
  
  card.addSection(section);
  
  return card.build();
}

/**
 * Analyzes the email using the Sentinel backend API
 * This function will be triggered when the "Analyze for Phishing" button is clicked
 * 
 * @param {Object} e - Event object containing email data as parameters
 * @return {ActionResponse} Updated card with analysis results
 */
function analyzeEmail(e) {
  var params = e.parameters;
  var subject = (params.subject || '').trim();
  var from = (params.from || '').trim();
  var body = (params.body || '').trim();
  
  try {
    // Ensure BACKEND_URL is clean
    var backendUrl = CONFIG.BACKEND_URL.trim();
    Logger.log("Sending request to: " + backendUrl);
    Logger.log("Subject: " + subject);
    Logger.log("From: " + from);
    Logger.log("Body length: " + body.length);
    
    // Prepare the request payload
    var payload = {
      'text': body,
      'subject': subject
    };
    
    var options = {
      'method': 'post',
      'contentType': 'application/json',
      'payload': JSON.stringify(payload),
      'muteHttpExceptions': true,
      'timeout': 60  // Google Apps Script max practical timeout is ~30-60 seconds
    };
    
    // Call the backend API
    var response = UrlFetchApp.fetch(backendUrl, options);
    var responseCode = response.getResponseCode();
    
    Logger.log("Response code: " + responseCode);
    
    if (responseCode === 200) {
      var result = JSON.parse(response.getContentText());
      return buildAnalysisResultCard(subject, from, result);
    } else {
      var errorText = response.getContentText();
      Logger.log("Backend error response: " + errorText);
      return buildErrorCard('Backend returned error: ' + responseCode + '\n' + errorText);
    }
    
  } catch (error) {
    Logger.log('Error calling backend: ' + error.toString());
    Logger.log('Error type: ' + error.name);
    
    // Specific handling for timeout errors
    if (error.toString().indexOf('timed out') > -1) {
      return buildErrorCard('⏱️ Analysis took too long (timeout).\n\n⚠️ This means the backend is still loading models.\n\n✅ Solution:\n1. Wait 30 seconds\n2. Close and reopen this email\n3. The models should be loaded by then\n\nNote: First request after server restart takes 30-60 seconds for model loading.');
    }
    
    return buildErrorCard('Unable to connect to backend: ' + error.toString() + '\n\nMake sure:\n1. Backend is running on port 8000\n2. Use ngrok for localhost exposure\n3. Check internet connection');
  }
}

/**
 * Builds a card showing the phishing analysis results
 * 
 * @param {string} subject - Email subject
 * @param {string} from - Email sender
 * @param {Object} analysis - Analysis result from backend
 * @return {ActionResponse} Card with analysis results
 */
function buildAnalysisResultCard(subject, from, analysis) {
  var card = CardService.newCardBuilder();
  
  // Determine header based on label and human intervention flag
  var headerTitle, headerSubtitle;
  
  if (analysis.label === 'uncertain') {
    headerTitle = '🔍 HUMAN REVIEW NEEDED';
    headerSubtitle = 'Models are in disagreement';
  } else if (analysis.label === 'phish') {
    headerTitle = '⚠️ PHISHING DETECTED';
    headerSubtitle = 'Confidence: ' + (analysis.confidence * 100).toFixed(0) + '%';
  } else {
    headerTitle = '✅ Email Appears Safe';
    headerSubtitle = 'Confidence: ' + (analysis.confidence * 100).toFixed(0) + '%';
  }
  
  var header = CardService.newCardHeader()
    .setTitle(headerTitle)
    .setSubtitle(headerSubtitle);
  
  card.setHeader(header);
  
  // Email info section
  var emailSection = CardService.newCardSection();
  emailSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Subject')
      .setContent(subject)
      .setMultiline(true)
  );
  emailSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('From')
      .setContent(from)
      .setMultiline(true)
  );
  card.addSection(emailSection);
  
  // Analysis section
  var analysisSection = CardService.newCardSection();
  
  // If uncertain, show both model scores for transparency
  if (analysis.label === 'uncertain') {
    analysisSection.addWidget(
      CardService.newTextParagraph()
        .setText('<b>Model Scores:</b><br>' +
                'NLP Model (Deberta): ' + (analysis.deberta_score * 100).toFixed(0) + '%<br>' +
                'SLM Model (Gemma): ' + (analysis.gemma_score * 100).toFixed(0) + '%')
    );
    analysisSection.addWidget(CardService.newDivider());
    analysisSection.addWidget(
      CardService.newTextParagraph()
        .setText('<b>⚠️ Alert:</b> The two detection models are giving conflicting signals about this email. ' +
                'Please review it carefully before taking action. Look for suspicious links, unexpected attachments, or unusual requests.')
    );
  } else {
    // Tactics for non-uncertain cases
    if (analysis.tactics && analysis.tactics.length > 0) {
      analysisSection.addWidget(
        CardService.newTextParagraph()
          .setText('<b>Detected Tactics:</b><br>' + analysis.tactics.join(', '))
      );
    }
  }
  
  // User tip
  if (analysis.user_tip) {
    analysisSection.addWidget(CardService.newDivider());
    analysisSection.addWidget(
      CardService.newTextParagraph()
        .setText('<b>💡 Security Tip:</b><br>' + analysis.user_tip)
    );
  }
  
  // Explanation
  if (analysis.explanation) {
    analysisSection.addWidget(
      CardService.newTextParagraph()
        .setText('<b>📋 Analysis:</b><br>' + analysis.explanation)
    );
  }
  
  card.addSection(analysisSection);
  
  // Build and return the action response
  var navigation = CardService.newNavigation().updateCard(card.build());
  return CardService.newActionResponseBuilder()
    .setNavigation(navigation)
    .build();
}

/**
 * Builds an error card
 * 
 * @param {string} errorMessage - Error message to display
 * @return {ActionResponse} Error card
 */
function buildErrorCard(errorMessage) {
  var card = CardService.newCardBuilder();
  
  var header = CardService.newCardHeader()
    .setTitle('Error')
    .setSubtitle('Analysis Failed');
  
  card.setHeader(header);
  
  var section = CardService.newCardSection();
  section.addWidget(
    CardService.newTextParagraph()
      .setText(errorMessage)
  );
  
  card.addSection(section);
  
  var navigation = CardService.newNavigation().updateCard(card.build());
  return CardService.newActionResponseBuilder()
    .setNavigation(navigation)
    .build();
}
