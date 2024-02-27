module com.example.prova3 {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;

    opens com.example.prova3 to javafx.fxml;
    exports com.example.prova3;
}